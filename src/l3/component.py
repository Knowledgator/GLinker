from typing import List
from gliner import GLiNER
from src.core.base import BaseComponent
from src.core.registry import component_registry
from .models import L3Config, L3Entity


class L3BaseComponent(BaseComponent[L3Config]):
    def _setup(self):
        self.model = GLiNER.from_pretrained(
            self.config.model_name,
            token=self.config.token
        )
        self.model.to(self.config.device)
    
    def get_available_methods(self) -> List[str]:
        return [
            "predict_entities",
            "filter_by_score",
            "sort_by_position",
            "deduplicate_entities"
        ]
    
    def predict_entities(self, text: str, labels: List[str]) -> List[L3Entity]:
        if not labels:
            return []
        
        entities = self.model.predict_entities(
            text,
            labels,
            threshold=self.config.threshold,
            flat_ner=self.config.flat_ner,
            multi_label=self.config.multi_label
        )
        
        return [
            L3Entity(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                score=e["score"]
            )
            for e in entities
        ]
    
    def filter_by_score(self, entities: List[L3Entity], threshold: float = None) -> List[L3Entity]:
        threshold = threshold if threshold is not None else self.config.threshold
        return [e for e in entities if e.score >= threshold]
    
    def sort_by_position(self, entities: List[L3Entity]) -> List[L3Entity]:
        return sorted(entities, key=lambda e: e.start)
    
    def deduplicate_entities(self, entities: List[L3Entity]) -> List[L3Entity]:
        seen = set()
        unique = []
        for entity in entities:
            key = (entity.text, entity.start, entity.end)
            if key not in seen:
                unique.append(entity)
                seen.add(key)
        return unique


@component_registry.register("l3_gliner")
class L3GLiNERComponent(L3BaseComponent):
    pass


@component_registry.register("l3_batch")
class L3BatchGLiNERComponent(L3BaseComponent):
    pass


@component_registry.register("l3_strict")
class L3StrictGLiNERComponent(L3BaseComponent):
    def predict_entities(self, text: str, labels: List[str]) -> List[L3Entity]:
        if not labels:
            return []
        
        entities = self.model.predict_entities(
            text,
            labels,
            threshold=self.config.threshold,
            flat_ner=True,
            multi_label=True
        )
        
        return [
            L3Entity(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                score=e["score"]
            )
            for e in entities
        ]