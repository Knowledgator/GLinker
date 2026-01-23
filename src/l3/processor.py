from typing import Any, List
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
                labels = self._create_gliner_labels(text_candidates)
            else:
                labels = [self._extract_label(c) for c in text_candidates]
            
            # Predict entities
            entities = self.component.predict_entities(text, labels)
            
            # Apply rest of pipeline
            for method_name, kwargs in self.pipeline[1:]:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)
            
            # Apply ranking if configured
            if self.schema.get('ranking'):
                entities = self._rank_entities(entities, text_candidates)
            
            all_entities.append(entities)
        
        return L3Output(entities=all_entities)
    
    def _extract_label(self, candidate: Any) -> str:
        """Extract label from candidate"""
        if hasattr(candidate, 'label'):
            return candidate.label
        return str(candidate)
    
    def _create_gliner_labels(self, candidates: List[Any]) -> List[str]:
        """Create GLiNER labels using schema template"""
        template = self.schema.get('template', '{label}')
        labels = []
        seen = set()
        
        for candidate in candidates:
            try:
                if hasattr(candidate, 'dict'):
                    cand_dict = candidate.dict()
                elif isinstance(candidate, dict):
                    cand_dict = candidate
                else:
                    labels.append(str(candidate))
                    continue
                
                label = template.format(**cand_dict)
                label_lower = label.lower()
                if label_lower not in seen:
                    labels.append(label)
                    seen.add(label_lower)
            except (KeyError, AttributeError):
                if hasattr(candidate, 'label'):
                    if candidate.label.lower() not in seen:
                        labels.append(candidate.label)
                        seen.add(candidate.label.lower())
        
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