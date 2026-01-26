"""
Tests for src/l3/component.py - L3 GLiNER component.
"""

import pytest


class TestL3ComponentCreation:
    """Tests for L3Component initialization."""

    def test_import(self):
        from glinker.l3.component import L3Component
        assert L3Component is not None

    def test_creation(self, l3_component):
        assert l3_component is not None

    def test_has_model(self, l3_component):
        assert l3_component.model is not None

    def test_has_config(self, l3_component):
        assert l3_component.config is not None

    def test_device_property(self, l3_component):
        assert l3_component.device == "cpu"

    def test_get_available_methods(self, l3_component):
        methods = l3_component.get_available_methods()
        assert isinstance(methods, list)
        assert "predict_entities" in methods
        assert "predict_with_embeddings" in methods
        assert "filter_by_score" in methods
        assert "sort_by_position" in methods
        assert "deduplicate_entities" in methods


class TestL3ComponentPrecomputedEmbeddings:
    """Tests for precomputed embeddings support."""

    def test_supports_precomputed_embeddings(self, l3_component):
        # BiEncoder models should support precomputed embeddings
        assert hasattr(l3_component, 'supports_precomputed_embeddings')
        # The actual value depends on the model

    def test_encode_labels_exists(self, l3_component):
        assert hasattr(l3_component, 'encode_labels')


class TestL3ComponentPredictEntities:
    """Tests for predict_entities method."""

    def test_predict_simple(self, l3_component):
        text = "TP53 is a gene."
        labels = ["gene", "disease", "protein"]
        entities = l3_component.predict_entities(text, labels)
        assert isinstance(entities, list)

    def test_predict_returns_l3entity(self, l3_component):
        from glinker.l3.models import L3Entity
        text = "BRCA1 mutations cause breast cancer."
        labels = ["gene", "disease"]
        entities = l3_component.predict_entities(text, labels)
        for entity in entities:
            assert isinstance(entity, L3Entity)

    def test_predict_empty_labels(self, l3_component):
        entities = l3_component.predict_entities("Some text", [])
        assert entities == []

    def test_entity_has_all_fields(self, l3_component):
        text = "TP53 is important."
        labels = ["gene"]
        entities = l3_component.predict_entities(text, labels)
        for entity in entities:
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'label')
            assert hasattr(entity, 'start')
            assert hasattr(entity, 'end')
            assert hasattr(entity, 'score')

    def test_entity_positions_valid(self, l3_component):
        text = "BRCA1 causes breast cancer."
        labels = ["gene", "disease"]
        entities = l3_component.predict_entities(text, labels)
        for entity in entities:
            assert entity.start >= 0
            assert entity.end > entity.start
            assert entity.end <= len(text)


class TestL3ComponentFilterByScore:
    """Tests for filter_by_score method."""

    def test_filter_by_score(self, l3_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="A", label="X", start=0, end=1, score=0.9),
            L3Entity(text="B", label="X", start=5, end=6, score=0.4),
            L3Entity(text="C", label="X", start=10, end=11, score=0.6),
        ]
        filtered = l3_component.filter_by_score(entities, threshold=0.5)
        assert len(filtered) == 2
        assert all(e.score >= 0.5 for e in filtered)

    def test_filter_by_score_default_threshold(self, l3_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="A", label="X", start=0, end=1, score=0.9),
            L3Entity(text="B", label="X", start=5, end=6, score=0.1),
        ]
        # Should use config threshold (0.3)
        filtered = l3_component.filter_by_score(entities)
        assert len(filtered) == 1

    def test_filter_empty(self, l3_component):
        filtered = l3_component.filter_by_score([])
        assert filtered == []


class TestL3ComponentSortByPosition:
    """Tests for sort_by_position method."""

    def test_sort_by_position(self, l3_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="C", label="X", start=20, end=21, score=0.9),
            L3Entity(text="A", label="X", start=0, end=1, score=0.8),
            L3Entity(text="B", label="X", start=10, end=11, score=0.7),
        ]
        sorted_ents = l3_component.sort_by_position(entities)
        assert sorted_ents[0].text == "A"
        assert sorted_ents[1].text == "B"
        assert sorted_ents[2].text == "C"

    def test_sort_empty(self, l3_component):
        sorted_ents = l3_component.sort_by_position([])
        assert sorted_ents == []


class TestL3ComponentDeduplicate:
    """Tests for deduplicate_entities method."""

    def test_deduplicate(self, l3_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.9),
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.85),
            L3Entity(text="BRCA1", label="gene", start=10, end=15, score=0.8),
        ]
        deduped = l3_component.deduplicate_entities(entities)
        assert len(deduped) == 2

    def test_deduplicate_keeps_first(self, l3_component):
        from glinker.l3.models import L3Entity
        entities = [
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.9),
            L3Entity(text="TP53", label="gene", start=0, end=4, score=0.5),
        ]
        deduped = l3_component.deduplicate_entities(entities)
        assert len(deduped) == 1
        assert deduped[0].score == 0.9  # Keeps first one

    def test_deduplicate_empty(self, l3_component):
        deduped = l3_component.deduplicate_entities([])
        assert deduped == []
