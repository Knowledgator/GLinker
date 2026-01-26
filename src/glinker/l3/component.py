from typing import List
import torch
from gliner import GLiNER
from glinker.core.base import BaseComponent
from .models import L3Config, L3Entity


class L3Component(BaseComponent[L3Config]):
    """GLiNER-based entity linking component"""

    def _setup(self):
        """Initialize GLiNER model"""
        self.model = GLiNER.from_pretrained(
            self.config.model_name,
            token=self.config.token,
            max_length=self.config.max_length
        )
        self.model.to(self.config.device)

        # Fix labels tokenizer max_length for BiEncoder models
        # Some models have model_max_length not properly set (> 10^18)
        if (self.config.max_length is not None and
            hasattr(self.model, 'data_processor') and
            hasattr(self.model.data_processor, 'labels_tokenizer')):
            tok = self.model.data_processor.labels_tokenizer
            if tok.model_max_length > 100000:
                tok.model_max_length = self.config.max_length

    @property
    def device(self):
        return self.config.device

    @property
    def supports_precomputed_embeddings(self) -> bool:
        """Check if model supports precomputed embeddings (BiEncoder)"""
        return hasattr(self.model, 'encode_labels') and self.model.config.labels_encoder is not None

    def get_available_methods(self) -> List[str]:
        return [
            "predict_entities",
            "predict_with_embeddings",
            "encode_labels",
            "filter_by_score",
            "sort_by_position",
            "deduplicate_entities"
        ]

    def encode_labels(self, labels: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode labels using GLiNER's native label encoder.

        Args:
            labels: List of label strings to encode
            batch_size: Batch size for encoding

        Returns:
            Tensor of shape (num_labels, hidden_size)

        Raises:
            NotImplementedError: If model doesn't support label encoding (not BiEncoder)
        """
        if not self.supports_precomputed_embeddings:
            raise NotImplementedError(
                f"Model {self.config.model_name} doesn't support label precomputation. "
                "Only BiEncoder models support this feature."
            )

        return self.model.encode_labels(labels, batch_size=batch_size)

    def predict_with_embeddings(
        self,
        text: str,
        labels: List[str],
        embeddings: torch.Tensor
    ) -> List[L3Entity]:
        """
        Predict entities using pre-computed label embeddings.

        Args:
            text: Input text
            labels: List of label strings (for output mapping)
            embeddings: Pre-computed embeddings tensor (num_labels, hidden_size)

        Returns:
            List of L3Entity predictions
        """
        if not self.supports_precomputed_embeddings:
            # Fallback to regular prediction
            return self.predict_entities(text, labels)

        entities = self.model.predict_with_embeds(
            text,
            embeddings,
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