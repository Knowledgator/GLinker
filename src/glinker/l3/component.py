from typing import List

import torch

from glinker.core.base import BaseComponent
from glinker.core.gliner_model import GLinkerModel

from .models import L3Config, L3Entity


class L3Component(BaseComponent[L3Config]):
    """GLiNER-based entity linking component."""

    def _setup(self):
        self.model = GLinkerModel.from_pretrained(
            self.config.model_name,
            token=self.config.token,
            max_length=self.config.max_length,
        )
        self.model.to(self.config.device)

        # Fix labels tokenizer max_length for BiEncoder models.
        # label_max_length takes priority; falls back to max_length.
        _lbl_len = self.config.label_max_length or self.config.max_length
        if (
            _lbl_len is not None
            and hasattr(self.model, "data_processor")
            and hasattr(self.model.data_processor, "labels_tokenizer")
        ):
            tok = self.model.data_processor.labels_tokenizer
            if tok.model_max_length > 100_000:
                tok.model_max_length = _lbl_len

    @property
    def device(self):
        return self.config.device

    @property
    def supports_precomputed_embeddings(self) -> bool:
        return hasattr(self.model, "encode_labels") and self.model.config.labels_encoder is not None

    def get_available_methods(self) -> List[str]:
        return [
            "predict_entities",
            "predict_with_embeddings",
            "encode_labels",
            "filter_by_score",
            "sort_by_position",
            "deduplicate_entities",
        ]

    def encode_labels(self, labels: List[str], batch_size: int = 32) -> torch.Tensor:
        if not self.supports_precomputed_embeddings:
            raise NotImplementedError(
                f"Model {self.config.model_name} doesn't support label precomputation. "
                "Only BiEncoder models support this feature."
            )
        return self.model.encode_labels(labels, batch_size=batch_size)

    @property
    def _span_extraction_threshold(self) -> float:
        """BIO extraction threshold — falls back to threshold if not set."""
        return self.config.span_extraction_threshold or self.config.threshold

    def predict_entities(
        self,
        text: str,
        labels: List[str],
        input_spans: List[dict] | None = None,
        span_label_indices: List[List[int]] | None = None,
    ) -> List[L3Entity]:
        """Predict entities with optional sparse scoring.

        When both *input_spans* and *span_label_indices* are provided each
        mention is scored only against its own candidates (sparse path).
        Otherwise falls back to standard dense GLiNER scoring.
        """
        if not labels:
            return []
        entities = self.model.predict_entities(
            text,
            labels,
            span_label_indices=span_label_indices,
            input_spans=input_spans,
            threshold=self.config.threshold,
            span_extraction_threshold=self._span_extraction_threshold,
            flat_ner=self.config.flat_ner,
            multi_label=self.config.multi_label,
            return_class_probs=True,
        )
        return [self._to_entity(e) for e in entities]

    def predict_with_embeddings(
        self,
        text: str,
        labels: List[str],
        embeddings: torch.Tensor,
        input_spans: List[dict] | None = None,
        span_label_indices: List[List[int]] | None = None,
    ) -> List[L3Entity]:
        """Predict from pre-computed embeddings with optional sparse scoring."""
        if not labels:
            return []
        if not self.supports_precomputed_embeddings:
            return self.predict_entities(text, labels, input_spans, span_label_indices)

        entities = self.model.predict_with_embeds(
            text,
            embeddings,
            labels,
            span_label_indices=span_label_indices,
            input_spans=input_spans,
            threshold=self.config.threshold,
            span_extraction_threshold=self._span_extraction_threshold,
            flat_ner=self.config.flat_ner,
            multi_label=self.config.multi_label,
            return_class_probs=True,
        )
        return [self._to_entity(e) for e in entities]

    def filter_by_score(
        self, entities: List[L3Entity], threshold: float | None = None
    ) -> List[L3Entity]:
        threshold = threshold if threshold is not None else self.config.threshold
        return [e for e in entities if e.score >= threshold]

    def sort_by_position(self, entities: List[L3Entity]) -> List[L3Entity]:
        return sorted(entities, key=lambda e: e.start)

    def deduplicate_entities(self, entities: List[L3Entity]) -> List[L3Entity]:
        seen: set = set()
        unique = []
        for entity in entities:
            key = (entity.text, entity.start, entity.end)
            if key not in seen:
                unique.append(entity)
                seen.add(key)
        return unique

    @staticmethod
    def _to_entity(e: dict) -> L3Entity:
        return L3Entity(
            text=e["text"],
            label=e["label"],
            start=e["start"],
            end=e["end"],
            score=e["score"],
            class_probs=e.get("class_probs"),
        )
