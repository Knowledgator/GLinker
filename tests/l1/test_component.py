"""
Tests for src/l1/component.py - L1 NER component.
"""

import pytest


class TestL1ComponentCreation:
    """Tests for L1Component initialization."""

    def test_import(self):
        from src.l1.component import L1Component
        assert L1Component is not None

    def test_creation(self, l1_component):
        assert l1_component is not None

    def test_has_nlp(self, l1_component):
        assert l1_component.nlp is not None

    def test_has_config(self, l1_component):
        assert l1_component.config is not None

    def test_get_available_methods(self, l1_component):
        methods = l1_component.get_available_methods()
        assert isinstance(methods, list)
        assert "extract_entities" in methods


class TestL1ComponentExtractEntities:
    """Tests for extract_entities method."""

    def test_extract_simple(self, l1_component):
        text = "TP53 is a gene."
        entities = l1_component.extract_entities(text)
        assert isinstance(entities, list)

    def test_extract_returns_l1entity(self, l1_component):
        from src.l1.models import L1Entity
        text = "TP53 mutations cause cancer."
        entities = l1_component.extract_entities(text)
        for entity in entities:
            assert isinstance(entity, L1Entity)

    def test_extract_empty_text(self, l1_component):
        entities = l1_component.extract_entities("")
        assert entities == []

    def test_extract_no_entities(self, l1_component):
        text = "the a an is are"
        entities = l1_component.extract_entities(text)
        # May or may not find entities depending on model
        assert isinstance(entities, list)

    def test_entity_has_position(self, l1_component):
        text = "BRCA1 mutations are common."
        entities = l1_component.extract_entities(text)
        for entity in entities:
            assert hasattr(entity, 'start')
            assert hasattr(entity, 'end')
            assert entity.start >= 0
            assert entity.end > entity.start

    def test_entity_position_matches_text(self, l1_component):
        text = "TP53 is important."
        entities = l1_component.extract_entities(text)
        for entity in entities:
            extracted = text[entity.start:entity.end]
            assert extracted == entity.text

    def test_entity_has_context(self, l1_component):
        text = "The TP53 gene is important for research."
        entities = l1_component.extract_entities(text)
        for entity in entities:
            assert hasattr(entity, 'left_context')
            assert hasattr(entity, 'right_context')

    def test_entity_has_text(self, l1_component):
        text = "TP53 is a gene."
        entities = l1_component.extract_entities(text)
        for entity in entities:
            assert hasattr(entity, 'text')
            assert entity.text is not None


class TestL1ComponentMinEntityLength:
    """Tests for min_entity_length filtering."""

    def test_min_length_filtering(self, l1_config_dict):
        from src.l1.component import L1Component
        from src.l1.models import L1Config

        config = L1Config(**{**l1_config_dict, "min_entity_length": 4})
        component = L1Component(config)

        text = "BRCA1 and TP53 are genes"
        entities = component.extract_entities(text)
        # Apply filter
        filtered = component.filter_by_length(entities, min_length=4)

        for entity in filtered:
            assert len(entity.text) >= 4


class TestL1ComponentDeduplicate:
    """Tests for deduplicate method."""

    def test_deduplicate(self, l1_component):
        from src.l1.models import L1Entity

        entities = [
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context=""),
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context=""),
            L1Entity(text="BRCA1", start=10, end=15,
                     left_context="", right_context=""),
        ]

        deduped = l1_component.deduplicate(entities)
        assert len(deduped) == 2

    def test_deduplicate_empty(self, l1_component):
        deduped = l1_component.deduplicate([])
        assert deduped == []

    def test_deduplicate_no_duplicates(self, l1_component):
        from src.l1.models import L1Entity

        entities = [
            L1Entity(text="A", start=0, end=1,
                     left_context="", right_context=""),
            L1Entity(text="B", start=5, end=6,
                     left_context="", right_context=""),
        ]

        deduped = l1_component.deduplicate(entities)
        assert len(deduped) == 2


class TestL1ComponentSortByPosition:
    """Tests for sort_by_position method."""

    def test_sort_by_position(self, l1_component):
        from src.l1.models import L1Entity

        entities = [
            L1Entity(text="C", start=20, end=21,
                     left_context="", right_context=""),
            L1Entity(text="A", start=0, end=1,
                     left_context="", right_context=""),
            L1Entity(text="B", start=10, end=11,
                     left_context="", right_context=""),
        ]

        sorted_ents = l1_component.sort_by_position(entities)
        assert sorted_ents[0].text == "A"
        assert sorted_ents[1].text == "B"
        assert sorted_ents[2].text == "C"

    def test_sort_empty(self, l1_component):
        sorted_ents = l1_component.sort_by_position([])
        assert sorted_ents == []
