from typing import List
from gliner import GLiNER
from src.core.base import BaseComponent
from src.core.registry import component_registry
from .models import L3Config, L3Entity
from src.l2.models import L2Candidate


class L3BaseComponent(BaseComponent[L3Config]):
    """Base component for L3 - common utility methods"""
    
    def get_available_methods(self) -> List[str]:
        return [
            "predict_entities",
            "filter_by_score",
            "sort_by_position",
            "sort_by_score",
            "limit_entities"
        ]
    
    def filter_by_score(
        self,
        entities: List[L3Entity],
        min_score: float = None
    ) -> List[L3Entity]:
        """Filter entities by minimum score"""
        threshold = min_score if min_score is not None else self.config.threshold
        return [e for e in entities if e.score >= threshold]
    
    def sort_by_position(self, entities: List[L3Entity]) -> List[L3Entity]:
        """Sort entities by start position"""
        return sorted(entities, key=lambda x: x.start)
    
    def sort_by_score(self, entities: List[L3Entity]) -> List[L3Entity]:
        """Sort entities by confidence score (descending)"""
        return sorted(entities, key=lambda x: x.score, reverse=True)
    
    def limit_entities(
        self,
        entities: List[L3Entity],
        limit: int = 10
    ) -> List[L3Entity]:
        """Limit number of entities"""
        return entities[:limit]


@component_registry.register("l3_gliner")
class L3GLiNERComponent(L3BaseComponent):
    """GLiNER-based NER component"""
    
    def _setup(self):
        """Load GLiNER model"""
        print(f"Loading GLiNER model: {self.config.model_name}...")
        self.model = GLiNER.from_pretrained(self.config.model_name)
        
        if self.config.device != "cpu":
            import torch
            self.model = self.model.to(self.config.device)
        
        print("✓ Model loaded!")
    
    def predict_entities(
        self, 
        text: str, 
        candidates: List[L2Candidate],
        threshold: float = None
    ) -> List[L3Entity]:
        """
        Predict entities using L2 candidates as labels
        
        Args:
            text: Input text
            candidates: L2 candidates (their labels are used as GLiNER labels)
            threshold: Confidence threshold (optional)
        
        Returns:
            List of predicted entities (label = candidate name)
        """
        if not candidates:
            return []
        
        conf_threshold = threshold if threshold is not None else self.config.threshold
        
        labels = []
        seen = set()
        
        for candidate in candidates:
            label_lower = candidate.label.lower()
            if label_lower not in seen:
                labels.append(candidate.label)
                seen.add(label_lower)
        
        predictions = self.model.predict_entities(
            text, 
            labels,
            threshold=conf_threshold,
            flat_ner=self.config.flat_ner,
            multi_label=self.config.multi_label
        )
        
        entities = []
        for pred in predictions:
            entities.append(L3Entity(
                text=pred['text'],
                start=pred['start'],
                end=pred['end'],
                label=pred['label'],
                score=pred['score']
            ))
        
        return entities