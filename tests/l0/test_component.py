"""
Tests for src/l0/component.py - L0 aggregation component.
"""

import pytest


class TestL0ComponentCreation:
    """Tests for L0Component initialization."""

    def test_import(self):
        from glinker.l0.component import L0Component
        assert L0Component is not None

    def test_creation(self, l0_component):
        assert l0_component is not None

    def test_has_config(self, l0_component):
        assert l0_component.config is not None

    def test_get_available_methods(self, l0_component):
        methods = l0_component.get_available_methods()
        assert isinstance(methods, list)
        assert "aggregate" in methods
        assert "filter_by_confidence" in methods
        assert "sort_by_confidence" in methods
        assert "calculate_stats" in methods


class TestL0ComponentAggregate:
    """Tests for aggregate method."""

    def test_aggregate_simple(self, l0_component):
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context=" is a gene")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53", description="Tumor protein")
        ]]
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9)
        ]]

        result = l0_component.aggregate(l1_entities, l2_candidates, l3_entities)

        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0].mention_text == "TP53"

    def test_aggregate_empty(self, l0_component):
        result = l0_component.aggregate([], [], [])
        assert result == []

    def test_aggregate_no_l3_match(self, l0_component):
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53")
        ]]
        l3_entities = [[]]  # No L3 matches

        result = l0_component.aggregate(l1_entities, l2_candidates, l3_entities)

        assert len(result) == 1
        assert result[0][0].is_linked is False
        assert result[0][0].pipeline_stage == "l2_found"

    def test_aggregate_l1_only(self, l0_component):
        from glinker.l1.models import L1Entity

        l1_entities = [[
            L1Entity(text="UNKNOWN", start=0, end=7,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[]]  # No candidates
        l3_entities = [[]]  # No matches

        result = l0_component.aggregate(l1_entities, l2_candidates, l3_entities)

        assert len(result) == 1
        assert result[0][0].pipeline_stage == "l1_only"
        assert result[0][0].num_candidates == 0

    def test_aggregate_with_template(self, l0_component):
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53", description="Tumor protein")
        ]]
        l3_entities = [[
            L3Entity(text="TP53", label="TP53: Tumor protein", start=0, end=4, score=0.9)
        ]]

        # Use template matching
        result = l0_component.aggregate(
            l1_entities, l2_candidates, l3_entities,
            template="{label}: {description}"
        )

        assert result[0][0].is_linked is True


class TestL0ComponentPositionMatching:
    """Tests for position matching with tolerance."""

    def test_exact_position_match(self, l0_component):
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[DatabaseRecord(entity_id="1", label="TP53")]]
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9)
        ]]

        result = l0_component.aggregate(l1_entities, l2_candidates, l3_entities)
        assert result[0][0].is_linked is True

    def test_fuzzy_position_match_within_tolerance(self, l0_component):
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[DatabaseRecord(entity_id="1", label="TP53")]]
        # L3 position differs by 1 (within tolerance=2)
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=1, end=5, score=0.9)
        ]]

        result = l0_component.aggregate(l1_entities, l2_candidates, l3_entities)
        # Should still match due to fuzzy position matching
        assert result[0][0].is_linked is True

    def test_fuzzy_position_match_outside_tolerance(self, l0_config_dict):
        from glinker.l0.component import L0Component
        from glinker.l0.models import L0Config
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        # Create component with position_tolerance=1
        config = L0Config(**{**l0_config_dict, "position_tolerance": 1})
        component = L0Component(config)

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[DatabaseRecord(entity_id="1", label="TP53")]]
        # L3 position differs by 5 (outside tolerance=1)
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=5, end=9, score=0.9)
        ]]

        result = component.aggregate(l1_entities, l2_candidates, l3_entities)
        # Should NOT match due to position difference
        assert result[0][0].is_linked is False


class TestL0ComponentStrictMatching:
    """Tests for strict_matching mode."""

    def test_strict_matching_enabled(self, l0_config_dict):
        from glinker.l0.component import L0Component
        from glinker.l0.models import L0Config
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        config = L0Config(**{**l0_config_dict, "strict_matching": True})
        component = L0Component(config)

        # L1 finds "TP53" at 0-4
        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[DatabaseRecord(entity_id="1", label="TP53")]]
        # L3 finds "BRCA1" at 20-25 (not in L1)
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9),
            L3Entity(text="BRCA1", label="BRCA1", start=20, end=25, score=0.8)
        ]]

        result = component.aggregate(l1_entities, l2_candidates, l3_entities)

        # In strict mode, only L1 mentions should be included
        assert len(result[0]) == 1
        assert result[0][0].mention_text == "TP53"

    def test_strict_matching_disabled(self, l0_config_dict):
        from glinker.l0.component import L0Component
        from glinker.l0.models import L0Config
        from glinker.l1.models import L1Entity
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        config = L0Config(**{**l0_config_dict, "strict_matching": False})
        component = L0Component(config)

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53"),
            DatabaseRecord(entity_id="2", label="BRCA1")
        ]]
        # L3 finds additional entity not in L1
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9),
            L3Entity(text="BRCA1", label="BRCA1", start=20, end=25, score=0.8)
        ]]

        result = component.aggregate(l1_entities, l2_candidates, l3_entities)

        # In loose mode, L3-only entities should be included
        assert len(result[0]) == 2


class TestL0ComponentFilterByConfidence:
    """Tests for filter_by_confidence method."""

    def test_filter_by_confidence(self, l0_component):
        from glinker.l0.models import L0Entity, LinkedEntity

        entities = [[
            L0Entity(
                mention_text="A", mention_start=0, mention_end=1,
                left_context="", right_context="",
                linked_entity=LinkedEntity(
                    entity_id="1", label="A", confidence=0.9,
                    start=0, end=1, matched_text="A"
                ),
                is_linked=True
            ),
            L0Entity(
                mention_text="B", mention_start=5, mention_end=6,
                left_context="", right_context="",
                linked_entity=LinkedEntity(
                    entity_id="2", label="B", confidence=0.3,
                    start=5, end=6, matched_text="B"
                ),
                is_linked=True
            ),
        ]]

        filtered = l0_component.filter_by_confidence(entities, min_confidence=0.5)
        assert len(filtered[0]) == 1
        assert filtered[0][0].mention_text == "A"

    def test_filter_by_confidence_empty(self, l0_component):
        filtered = l0_component.filter_by_confidence([[]])
        assert filtered == [[]]


class TestL0ComponentSortByConfidence:
    """Tests for sort_by_confidence method."""

    def test_sort_by_confidence(self, l0_component):
        from glinker.l0.models import L0Entity, LinkedEntity

        entities = [[
            L0Entity(
                mention_text="B", mention_start=5, mention_end=6,
                left_context="", right_context="",
                linked_entity=LinkedEntity(
                    entity_id="2", label="B", confidence=0.5,
                    start=5, end=6, matched_text="B"
                ),
                is_linked=True
            ),
            L0Entity(
                mention_text="A", mention_start=0, mention_end=1,
                left_context="", right_context="",
                linked_entity=LinkedEntity(
                    entity_id="1", label="A", confidence=0.9,
                    start=0, end=1, matched_text="A"
                ),
                is_linked=True
            ),
        ]]

        sorted_ents = l0_component.sort_by_confidence(entities)
        assert sorted_ents[0][0].mention_text == "A"  # Higher confidence first
        assert sorted_ents[0][1].mention_text == "B"


class TestL0ComponentCalculateStats:
    """Tests for calculate_stats method."""

    def test_calculate_stats(self, l0_component):
        from glinker.l0.models import L0Entity, LinkedEntity

        entities = [[
            L0Entity(
                mention_text="A", mention_start=0, mention_end=1,
                left_context="", right_context="",
                linked_entity=LinkedEntity(
                    entity_id="1", label="A", confidence=0.9,
                    start=0, end=1, matched_text="A"
                ),
                is_linked=True,
                pipeline_stage="l3_linked"
            ),
            L0Entity(
                mention_text="B", mention_start=5, mention_end=6,
                left_context="", right_context="",
                is_linked=False,
                pipeline_stage="l1_only"
            ),
        ]]

        stats = l0_component.calculate_stats(entities)

        assert stats["total_mentions"] == 2
        assert stats["linked"] == 1
        assert stats["unlinked"] == 1
        assert stats["linking_rate"] == 0.5
        assert stats["stages"]["l3_linked"] == 1
        assert stats["stages"]["l1_only"] == 1

    def test_calculate_stats_empty(self, l0_component):
        stats = l0_component.calculate_stats([])
        assert stats["total_mentions"] == 0
        assert stats["linking_rate"] == 0.0
