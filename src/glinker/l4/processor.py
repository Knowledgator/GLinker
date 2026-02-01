from typing import Any, List, Optional
from glinker.core.base import BaseProcessor
from glinker.core.registry import processor_registry
from glinker.l3.models import L3Input, L3Output, L3Entity
from .models import L4Config
from .component import L4Component


class L4Processor(BaseProcessor[L4Config, L3Input, L3Output]):
    """GLiNER reranking processor with candidate chunking"""

    def __init__(
        self,
        config: L4Config,
        component: L4Component,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
        self._validate_pipeline()
        self.schema = {}

    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("predict_entities_chunked", {}),
            ("deduplicate_entities", {}),
            ("filter_by_score", {}),
            ("sort_by_position", {})
        ]

    @staticmethod
    def _build_input_spans(l1_entities_for_text: List[Any]) -> List[List[dict]]:
        """Convert L1 entities to GLiNER input_spans format."""
        spans = [{"start": e.start, "end": e.end} for e in l1_entities_for_text]
        return [spans]

    def __call__(
        self,
        texts: List[str] = None,
        candidates: List[List[Any]] = None,
        l1_entities: List[List[Any]] = None,
        input_data: L3Input = None
    ) -> L3Output:
        """Process texts with candidate labels using chunked GLiNER inference.

        Args:
            texts: List of input texts
            candidates: List of candidate lists per text (from L2)
            l1_entities: Optional L1 entities per text, used to build input_spans
            input_data: Alternative L3Input object
        """
        if texts is not None and candidates is not None:
            texts_to_process = texts
            candidates_to_process = candidates
        elif input_data is not None:
            texts_to_process = input_data.texts
            candidates_to_process = input_data.labels
        else:
            raise ValueError("Either 'texts'+'candidates' or 'input_data' must be provided")

        all_entities = []
        max_labels = self.config.max_labels

        # Detect shared candidates (all texts use the same list)
        shared = (
            len(candidates_to_process) > 1
            and all(c is candidates_to_process[0] for c in candidates_to_process[1:])
        )

        shared_labels = None
        if shared:
            ref_candidates = candidates_to_process[0]
            if self.schema:
                shared_labels, _ = self._create_gliner_labels_with_mapping(ref_candidates)
            else:
                shared_labels = [self._extract_label(c) for c in ref_candidates]

        for idx, (text, text_candidates) in enumerate(zip(texts_to_process, candidates_to_process)):
            # Build input_spans from L1 entities if available
            input_spans = None
            if l1_entities is not None and idx < len(l1_entities):
                text_l1 = l1_entities[idx]
                if text_l1:
                    input_spans = self._build_input_spans(text_l1)

            if shared:
                labels = shared_labels
            else:
                if self.schema:
                    labels, _ = self._create_gliner_labels_with_mapping(text_candidates)
                else:
                    labels = [self._extract_label(c) for c in text_candidates]

            # Run chunked prediction
            entities = self.component.predict_entities_chunked(
                text, labels, max_labels, input_spans=input_spans
            )

            # Apply remaining pipeline steps (deduplicate, filter, sort)
            for method_name, kwargs in self.pipeline[1:]:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)

            all_entities.append(entities)

        return L3Output(entities=all_entities)

    def _extract_label(self, candidate: Any) -> str:
        """Extract label from candidate"""
        if hasattr(candidate, 'label'):
            return candidate.label
        return str(candidate)

    def _create_gliner_labels_with_mapping(self, candidates: List[Any]) -> tuple:
        """Create GLiNER labels using schema template and return label->candidate mapping."""
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


@processor_registry.register("l4_reranker")
def create_l4_processor(config_dict: dict, pipeline: list = None) -> L4Processor:
    """Factory: creates component + processor"""
    config = L4Config(**config_dict)
    component = L4Component(config)
    return L4Processor(config, component, pipeline)