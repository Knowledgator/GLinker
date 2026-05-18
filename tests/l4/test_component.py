"""
Tests for src/l4/component.py - L4 GLiNER reranking component.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestL4ComponentCreation:
    """Tests for L4Component initialization."""

    def test_import(self):
        from glinker.l4.component import L4Component
        assert L4Component is not None

    def test_creation(self, l4_component):
        assert l4_component is not None

    def test_has_model(self, l4_component):
        assert l4_component.model is not None

    def test_has_config(self, l4_component):
        assert l4_component.config is not None

    def test_device_property(self, l4_component):
        # Should be on CPU for tests
        assert hasattr(l4_component.config, 'device')

    def test_get_available_methods(self, l4_component):
        methods = l4_component.get_available_methods()
        assert isinstance(methods, list)
        assert "predict_entities" in methods
        assert "predict_entities_chunked" in methods
        assert "filter_by_score" in methods
        assert "sort_by_position" in methods
        assert "deduplicate_entities" in methods


class TestL4ComponentPredictEntities:
    """Tests for predict_entities method."""

    def test_predict_simple(self, l4_component):
        text = "TP53 is a gene."
        labels = ["gene", "disease", "protein"]
        entities = l4_component.predict_entities(text, labels)
        assert isinstance(entities, list)

    def test_predict_returns_l3entity(self, l4_component):
        from glinker.l3.models import L3Entity
        text = "BRCA1 mutations cause breast cancer."
        labels = ["gene", "disease"]
        entities = l4_component.predict_entities(text, labels)
        for entity in entities:
            assert isinstance(entity, L3Entity)

    def test_predict_empty_labels(self, l4_component):
        entities = l4_component.predict_entities("Some text", [])
        assert entities == []

    def test_predict_with_input_spans(self, l4_component):
        text = "TP53 mutations cause cancer."
        labels = ["gene", "disease"]
        # Provide spans to constrain prediction
        input_spans = [[{"start": 0, "end": 4}]]  # Only "TP53"
        entities = l4_component.predict_entities(
            text, labels, input_spans=input_spans
        )
        assert isinstance(entities, list)
        # If entities found, they should be within input_spans
        for entity in entities:
            assert entity.start >= 0
            assert entity.end <= len(text)

    def test_entity_has_all_fields(self, l4_component):
        text = "TP53 is important."
        labels = ["gene"]
        entities = l4_component.predict_entities(text, labels)
        for entity in entities:
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'label')
            assert hasattr(entity, 'start')
            assert hasattr(entity, 'end')
            assert hasattr(entity, 'score')

    def test_entity_positions_valid(self, l4_component):
        text = "BRCA1 causes breast cancer."
        labels = ["gene", "disease"]
        entities = l4_component.predict_entities(text, labels)
        for entity in entities:
            assert entity.start >= 0
            assert entity.end > entity.start
            assert entity.end <= len(text)

    def test_predict_with_class_probs(self, l4_component):
        """Verify that class_probs are returned when requested."""
        text = "TP53 mutations."
        labels = ["gene", "protein"]
        entities = l4_component.predict_entities(text, labels)
        for entity in entities:
            # class_probs should be included (return_class_probs=True)
            assert hasattr(entity, 'class_probs')


class TestL4ComponentPredictEntitiesChunked:
    """Tests for predict_entities_chunked method."""

    def test_predict_chunked_small_labels(self, l4_component):
        """When labels <= max_labels, should behave like predict_entities."""
        text = "TP53 and BRCA1 are genes."
        labels = ["gene", "disease"]
        max_labels = 10
        entities = l4_component.predict_entities_chunked(
            text, labels, max_labels
        )
        assert isinstance(entities, list)

    def test_predict_chunked_large_labels(self, l4_component):
        """When labels > max_labels, should split into chunks."""
        text = "TP53 is a gene."
        # Create many labels to force chunking
        labels = [f"label_{i}" for i in range(50)]
        max_labels = 10
        entities = l4_component.predict_entities_chunked(
            text, labels, max_labels
        )
        assert isinstance(entities, list)

    def test_predict_chunked_empty_labels(self, l4_component):
        entities = l4_component.predict_entities_chunked(
            "Some text", [], max_labels=10
        )
        assert entities == []

    def test_predict_chunked_with_input_spans(self, l4_component):
        text = "TP53 mutations cause cancer."
        labels = ["gene", "disease", "protein"]
        input_spans = [[{"start": 0, "end": 4}]]
        entities = l4_component.predict_entities_chunked(
            text, labels, max_labels=2, input_spans=input_spans
        )
        assert isinstance(entities, list)

    def test_predict_chunked_exact_boundary(self, l4_component):
        """Test when len(labels) == max_labels."""
        text = "Test text."
        labels = ["A", "B", "C"]
        entities = l4_component.predict_entities_chunked(
            text, labels, max_labels=3
        )
        assert isinstance(entities, list)


class TestL4ComponentFilterByScore:
    """Tests for filter_by_score method."""

    def test_filter_by_score(self, l4_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="A", label="X", start=0, end=1, score=0.9),
            L3Entity(text="B", label="X", start=5, end=6, score=0.4),
            L3Entity(text="C", label="X", start=10, end=11, score=0.6),
        ]
        filtered = l4_component.filter_by_score(entities, threshold=0.5)
        assert len(filtered) == 2
        assert all(e.score >= 0.5 for e in filtered)

    def test_filter_by_score_default_threshold(self, l4_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="A", label="X", start=0, end=1, score=0.9),
            L3Entity(text="B", label="X", start=5, end=6, score=0.1),
        ]
        # Should use config threshold (0.5)
        filtered = l4_component.filter_by_score(entities)
        assert len(filtered) == 1

    def test_filter_empty(self, l4_component):
        filtered = l4_component.filter_by_score([])
        assert filtered == []


class TestL4ComponentSortByPosition:
    """Tests for sort_by_position method."""

    def test_sort_by_position(self, l4_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="C", label="X", start=20, end=21, score=0.9),
            L3Entity(text="A", label="X", start=0, end=1, score=0.8),
            L3Entity(text="B", label="X", start=10, end=11, score=0.7),
        ]
        sorted_ents = l4_component.sort_by_position(entities)
        assert sorted_ents[0].text == "A"
        assert sorted_ents[1].text == "B"
        assert sorted_ents[2].text == "C"

    def test_sort_empty(self, l4_component):
        sorted_ents = l4_component.sort_by_position([])
        assert sorted_ents == []


class TestL4ComponentDeduplicate:
    """Tests for deduplicate_entities method."""

    def test_deduplicate(self, l4_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.9),
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.85),
            L3Entity(text="BRCA1", label="gene", start=10, end=15, score=0.8),
        ]
        deduped = l4_component.deduplicate_entities(entities)
        assert len(deduped) == 2

    def test_deduplicate_keeps_highest_score(self, l4_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.5),
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.9),
        ]
        deduped = l4_component.deduplicate_entities(entities)
        assert len(deduped) == 1
        assert deduped[0].score == 0.9  # Keeps highest score

    def test_deduplicate_empty(self, l4_component):
        deduped = l4_component.deduplicate_entities([])
        assert deduped == []

    def test_deduplicate_different_spans(self, l4_component):
        """Different spans should not be deduplicated."""
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.9),
            L3Entity(text="TP53", label="gene", start=10, end=14, score=0.8),
        ]
        deduped = l4_component.deduplicate_entities(entities)
        assert len(deduped) == 2  # Different positions
