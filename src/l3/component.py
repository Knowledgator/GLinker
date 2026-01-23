from typing import List
from gliner import GLiNER
from src.core.base import BaseComponent
from .models import L3Config, L3Entity


class L3Component(BaseComponent[L3Config]):
    """GLiNER-based entity linking component"""
    
    def _setup(self):
        """Initialize GLiNER model"""
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
        """Predict entities using GLiNER"""
        if not labels:
            return []
        print(labels)
        # labels = [label[:75] for label in labels]  # Truncate long labels
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
        """Filter entities by confidence score"""
        threshold = threshold if threshold is not None else self.config.threshold
        return [e for e in entities if e.score >= threshold]
    
    def sort_by_position(self, entities: List[L3Entity]) -> List[L3Entity]:
        """Sort entities by position in text"""
        return sorted(entities, key=lambda e: e.start)
    
    def deduplicate_entities(self, entities: List[L3Entity]) -> List[L3Entity]:
        """Remove duplicate entities"""
        seen = set()
        unique = []
        for entity in entities:
            key = (entity.text, entity.start, entity.end)
            if key not in seen:
                unique.append(entity)
                seen.add(key)
        return unique