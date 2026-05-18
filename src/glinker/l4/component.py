from gliner import GLiNER

from glinker.core.base import BaseComponent
from glinker.l3.models import L3Entity

from .models import L4Config


class L4Component(BaseComponent[L4Config]):
    """GLiNER-based reranking component (uni-encoder only, no precomputed embeddings)."""

    def _setup(self):
        """Initialize GLiNER model."""
        self.model = GLiNER.from_pretrained(
            self.config.model_name, token=self.config.token, max_length=self.config.max_length
        )
        self.model.to(self.config.device)

    def get_available_methods(self) -> list[str]:
        return [
            "predict_entities",
            "predict_entities_chunked",
            "filter_by_score",
            "sort_by_position",
            "deduplicate_entities",
        ]

    def predict_entities(
        self, text: str, labels: list[str], input_spans: list[list[dict]] | None = None
    ) -> list[L3Entity]:
        """Predict entities using GLiNER for a single label set.

        Args:
            text: Input text
            labels: List of label strings
            input_spans: Optional list of span dicts with 'start' and 'end' keys
        """
        if not labels:
            return []

        kwargs = {
            "threshold": self.config.threshold,
            "flat_ner": self.config.flat_ner,
            "multi_label": self.config.multi_label,
            "return_class_probs": True,
        }
        if input_spans is not None:
            kwargs["input_spans"] = input_spans

        entities = self.model.predict_entities(text, labels, **kwargs)

        return [
            L3Entity(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                score=e["score"],
                class_probs=e.get("class_probs"),
            )
            for e in entities
        ]

    def predict_entities_chunked(
        self,
        text: str,
        labels: list[str],
        max_labels: int,
        input_spans: list[list[dict]] | None = None,
    ) -> list[L3Entity]:
        """Predict entities with candidate chunking.

        Splits labels into chunks of max_labels, runs inference on each chunk,
        and merges results.

        Args:
            text: Input text
            labels: Full list of candidate label strings
            max_labels: Maximum labels per inference call
            input_spans: Optional span constraints from L1 entities
        """
        if not labels:
            return []

        if len(labels) <= max_labels:
            return self.predict_entities(text, labels, input_spans=input_spans)

        # Split labels into chunks
        chunks = [labels[i : i + max_labels] for i in range(0, len(labels), max_labels)]

        all_entities = []
        for chunk in chunks:
            entities = self.predict_entities(text, chunk, input_spans=input_spans)
            all_entities.extend(entities)

        return all_entities

    def filter_by_score(
        self, entities: list[L3Entity], threshold: float | None = None
    ) -> list[L3Entity]:
        """Filter entities by confidence score."""
        threshold = threshold if threshold is not None else self.config.threshold
        return [e for e in entities if e.score >= threshold]

    def sort_by_position(self, entities: list[L3Entity]) -> list[L3Entity]:
        """Sort entities by position in text."""
        return sorted(entities, key=lambda e: e.start)

    def deduplicate_entities(self, entities: list[L3Entity]) -> list[L3Entity]:
        """Remove duplicate entities, keeping the highest-scoring one per span."""
        best = {}
        for entity in entities:
            key = (entity.text, entity.start, entity.end)
            if key not in best or entity.score > best[key].score:
                best[key] = entity
        return list(best.values())
