from typing import Any, List, Optional
import torch
from src.core.base import BaseProcessor
from src.core.registry import processor_registry
from .models import L3Config, L3Input, L3Output, L3Entity
from .component import L3Component


class L3Processor(BaseProcessor[L3Config, L3Input, L3Output]):
    """GLiNER entity linking processor"""

    def __init__(
        self,
        config: L3Config,
        component: L3Component,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
        self._validate_pipeline()
        self.schema = {}
        self._l2_processor = None  # Will be set by DAG executor for cache write-back
    
    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("predict_entities", {}),
            ("filter_by_score", {}),
            ("sort_by_position", {})
        ]
    
    def __call__(
        self,
        texts: List[str] = None,
        candidates: List[List[Any]] = None,
        input_data: L3Input = None
    ) -> L3Output:
        """Process texts with candidate labels"""

        # Support both direct params and L3Input
        if texts is not None and candidates is not None:
            texts_to_process = texts
            candidates_to_process = candidates
        elif input_data is not None:
            texts_to_process = input_data.texts
            candidates_to_process = input_data.labels
        else:
            raise ValueError("Either 'texts'+'candidates' or 'input_data' must be provided")

        all_entities = []

        for text, text_candidates in zip(texts_to_process, candidates_to_process):
            # Create labels from candidates
            if self.schema:
                labels, label_to_candidate = self._create_gliner_labels_with_mapping(text_candidates)
            else:
                labels = [self._extract_label(c) for c in text_candidates]
                label_to_candidate = {}

            # Check if we can use precomputed embeddings
            use_precomputed = (
                self.config.use_precomputed_embeddings
                and self.component.supports_precomputed_embeddings
                and self._can_use_precomputed(text_candidates, label_to_candidate)
            )

            if use_precomputed:
                # Get embeddings from candidates
                embeddings = self._get_embeddings_tensor(text_candidates, labels, label_to_candidate)
                entities = self.component.predict_with_embeddings(text, labels, embeddings)
            else:
                # Regular prediction
                entities = self.component.predict_entities(text, labels)

                # Optionally cache computed embeddings
                if self.config.cache_embeddings and self.component.supports_precomputed_embeddings:
                    self._cache_embeddings(text_candidates, labels, label_to_candidate)

            # Apply rest of pipeline
            for method_name, kwargs in self.pipeline[1:]:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)

            # Apply ranking if configured
            if self.schema.get('ranking'):
                entities = self._rank_entities(entities, text_candidates)

            all_entities.append(entities)

        return L3Output(entities=all_entities)

    def _can_use_precomputed(
        self,
        candidates: List[Any],
        label_to_candidate: dict
    ) -> bool:
        """Check if all candidates have compatible precomputed embeddings"""
        if not candidates:
            return False

        expected_model = self.config.model_name

        for candidate in candidates:
            # Check if candidate has embedding
            embedding = getattr(candidate, 'embedding', None)
            if embedding is None:
                return False

            # Check if model matches
            model_id = getattr(candidate, 'embedding_model_id', None)
            if model_id != expected_model:
                return False

        return True

    def _get_embeddings_tensor(
        self,
        candidates: List[Any],
        labels: List[str],
        label_to_candidate: dict
    ) -> torch.Tensor:
        """Build embeddings tensor from candidates in same order as labels"""
        embeddings = []

        for label in labels:
            candidate = label_to_candidate.get(label)
            if candidate and hasattr(candidate, 'embedding') and candidate.embedding:
                embeddings.append(candidate.embedding)
            else:
                # Should not happen if _can_use_precomputed returned True
                raise ValueError(f"Missing embedding for label: {label}")

        return torch.tensor(embeddings, device=self.component.device)

    def _cache_embeddings(
        self,
        candidates: List[Any],
        labels: List[str],
        label_to_candidate: dict
    ):
        """Compute and cache embeddings for candidates without them"""
        if not self._l2_processor:
            return

        # Find candidates without embeddings
        to_compute = []
        to_compute_ids = []

        for candidate in candidates:
            if not getattr(candidate, 'embedding', None):
                to_compute.append(candidate)
                to_compute_ids.append(candidate.entity_id)

        if not to_compute:
            return

        # Format labels for these candidates
        template = self.schema.get('template', '{label}')
        compute_labels = []
        for candidate in to_compute:
            try:
                if hasattr(candidate, 'model_dump'):
                    formatted = template.format(**candidate.model_dump())
                elif hasattr(candidate, 'dict'):
                    formatted = template.format(**candidate.dict())
                else:
                    formatted = candidate.label
                compute_labels.append(formatted)
            except KeyError:
                compute_labels.append(candidate.label)

        # Encode labels
        embeddings = self.component.encode_labels(compute_labels)

        # Update L2 layer
        if hasattr(self._l2_processor, 'component'):
            for layer in self._l2_processor.component.layers:
                if layer.is_available():
                    layer.update_embeddings(
                        to_compute_ids,
                        embeddings.tolist(),
                        self.config.model_name
                    )
                    break  # Update first available layer
    
    def _extract_label(self, candidate: Any) -> str:
        """Extract label from candidate"""
        if hasattr(candidate, 'label'):
            return candidate.label
        return str(candidate)

    def _create_gliner_labels_with_mapping(self, candidates: List[Any]) -> tuple:
        """
        Create GLiNER labels using schema template and return label->candidate mapping.

        Returns:
            tuple: (labels: List[str], label_to_candidate: dict)
        """
        template = self.schema.get('template', '{label}')
        labels = []
        label_to_candidate = {}
        seen = set()

        for candidate in candidates:
            try:
                if hasattr(candidate, 'model_dump'):
                    cand_dict = candidate.model_dump()
                elif hasattr(candidate, 'dict'):
                    cand_dict = candidate.dict()
                elif isinstance(candidate, dict):
                    cand_dict = candidate
                else:
                    label = str(candidate)
                    if label.lower() not in seen:
                        labels.append(label)
                        seen.add(label.lower())
                    continue

                label = template.format(**cand_dict)
                label_lower = label.lower()
                if label_lower not in seen:
                    labels.append(label)
                    label_to_candidate[label] = candidate
                    seen.add(label_lower)
            except (KeyError, AttributeError):
                if hasattr(candidate, 'label'):
                    if candidate.label.lower() not in seen:
                        labels.append(candidate.label)
                        label_to_candidate[candidate.label] = candidate
                        seen.add(candidate.label.lower())

        return labels, label_to_candidate

    def _create_gliner_labels(self, candidates: List[Any]) -> List[str]:
        """Create GLiNER labels using schema template (legacy, for compatibility)"""
        labels, _ = self._create_gliner_labels_with_mapping(candidates)
        return labels
    
    def _rank_entities(self, entities: List[L3Entity], candidates: List[Any]) -> List[L3Entity]:
        """Re-rank entities using multiple scoring factors"""
        # Build label to candidate mapping
        label_to_candidate = {}
        for c in candidates:
            if hasattr(c, 'label'):
                label_to_candidate[c.label] = c
                if hasattr(c, 'aliases'):
                    for alias in c.aliases:
                        if alias not in label_to_candidate:
                            label_to_candidate[alias] = c
        
        # Calculate weighted scores
        for entity in entities:
            total_score = 0.0
            total_weight = 0.0
            
            for rank_spec in self.schema['ranking']:
                field = rank_spec['field']
                weight = rank_spec['weight']
                total_weight += weight
                
                if field == 'gliner_score':
                    total_score += entity.score * weight
                else:
                    candidate = label_to_candidate.get(entity.label)
                    if candidate and hasattr(candidate, field):
                        value = getattr(candidate, field, 0)
                        if isinstance(value, (int, float)):
                            normalized = min(value / 1000000.0, 1.0)
                            total_score += normalized * weight
            
            if total_weight > 0:
                entity.score = total_score / total_weight
        
        return sorted(entities, key=lambda x: x.score, reverse=True)


@processor_registry.register("l3_batch")
def create_l3_processor(config_dict: dict, pipeline: list = None) -> L3Processor:
    """Factory: creates component + processor"""
    config = L3Config(**config_dict)
    component = L3Component(config)
    return L3Processor(config, component, pipeline)