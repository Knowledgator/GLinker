from typing import Any, List
from src.core.base import BaseProcessor
from src.core.registry import processor_registry, component_registry
from .models import L3Config, L3Input, L3Output, L3Entity
from .component import L3BaseComponent


class L3Processor(BaseProcessor[L3Config, L3Input, L3Output]):
    def __init__(
        self,
        config: L3Config,
        component: L3BaseComponent,
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
    
    def __call__(self, texts: List[str], candidates: List[List[Any]]) -> L3Output:
        all_entities = []
        
        for text, text_candidates in zip(texts, candidates):
            if self.schema:
                labels = self._create_gliner_labels(text_candidates)
            else:
                labels = [self._extract_label(c) for c in text_candidates]
            
            entities = self.component.predict_entities(text, labels)
            
            for method_name, kwargs in self.pipeline[1:]:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)
            
            if self.schema.get('ranking'):
                entities = self._rank_entities(entities, text_candidates)
            
            all_entities.append(entities)
        
        return L3Output(entities=all_entities)
    
    def _extract_label(self, candidate: Any) -> str:
        if hasattr(candidate, 'label'):
            return candidate.label
        return str(candidate)
    
    def _create_gliner_labels(self, candidates: List[Any]) -> List[str]:
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
    # TODO move to l0
    def _rank_entities(self, entities: List[L3Entity], candidates: List[Any]) -> List[L3Entity]:
        label_to_candidate = {}
        for c in candidates:
            if hasattr(c, 'label'):
                label_to_candidate[c.label] = c
                if hasattr(c, 'aliases'):
                    for alias in c.aliases:
                        if alias not in label_to_candidate:
                            label_to_candidate[alias] = c
        
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


@processor_registry.register("l3_default")
@processor_registry.register("l3_gliner")
def create_l3_gliner_processor(config_dict: dict, pipeline: list = None) -> L3Processor:
    config = L3Config(**config_dict)
    component_cls = component_registry.get("l3_gliner")
    component = component_cls(config)
    return L3Processor(config, component, pipeline)


@processor_registry.register("l3_batch")
def create_l3_batch_processor(config_dict: dict, pipeline: list = None) -> L3Processor:
    config = L3Config(**config_dict)
    component_cls = component_registry.get("l3_batch")
    component = component_cls(config)
    return L3Processor(config, component, pipeline)


@processor_registry.register("l3_strict")
def create_l3_strict_processor(config_dict: dict, pipeline: list = None) -> L3Processor:
    config = L3Config(**config_dict)
    component_cls = component_registry.get("l3_strict")
    component = component_cls(config)
    return L3Processor(config, component, pipeline)