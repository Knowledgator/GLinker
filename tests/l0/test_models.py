"""
Tests for src/l0/models.py - L0 aggregation data models.
"""

import pytest
from pydantic import ValidationError


class TestL0Config:
    """Tests for L0Config."""

    def test_import(self):
        from src.l0.models import L0Config
        assert L0Config is not None

    def test_creation(self, l0_config_dict):
        from src.l0.models import L0Config
        config = L0Config(**l0_config_dict)
        assert config.min_confidence == 0.0
        assert config.strict_matching is True
        assert config.position_tolerance == 2

    def test_default_values(self):
        from src.l0.models import L0Config
        config = L0Config()
        assert config.min_confidence == 0.0
        assert config.include_unlinked is True
        assert config.return_all_candidates is False
        assert config.strict_matching is True
        assert config.position_tolerance == 2

    def test_custom_position_tolerance(self):
        from src.l0.models import L0Config
        config = L0Config(position_tolerance=5)
        assert config.position_tolerance == 5


class TestL0Input:
    """Tests for L0Input."""

    def test_import(self):
        from src.l0.models import L0Input
        assert L0Input is not None

    def test_creation(self):
        from src.l0.models import L0Input
        from src.l1.models import L1Entity
        from src.l2.models import DatabaseRecord
        from src.l3.models import L3Entity

        l1_entity = L1Entity(
            text="TP53", start=0, end=4, label="GENE",
            left_context="", right_context=""
        )
        l2_record = DatabaseRecord(entity_id="1", label="TP53")
        l3_entity = L3Entity(
            text="TP53", label="TP53", start=0, end=4, score=0.9
        )

        input_data = L0Input(
            l1_entities=[[l1_entity]],
            l2_candidates=[[l2_record]],
            l3_entities=[[l3_entity]]
        )

        assert len(input_data.l1_entities) == 1
        assert len(input_data.l2_candidates) == 1
        assert len(input_data.l3_entities) == 1


class TestLinkedEntity:
    """Tests for LinkedEntity."""

    def test_import(self):
        from src.l0.models import LinkedEntity
        assert LinkedEntity is not None

    def test_creation(self):
        from src.l0.models import LinkedEntity
        linked = LinkedEntity(
            entity_id="MESH:123",
            label="TP53",
            confidence=0.95,
            start=0,
            end=4,
            matched_text="TP53"
        )
        assert linked.entity_id == "MESH:123"
        assert linked.label == "TP53"
        assert linked.confidence == 0.95

    def test_all_fields_required(self):
        from src.l0.models import LinkedEntity
        with pytest.raises(ValidationError):
            LinkedEntity(entity_id="1", label="TP53")  # Missing confidence, positions


class TestL0Entity:
    """Tests for L0Entity."""

    def test_import(self):
        from src.l0.models import L0Entity
        assert L0Entity is not None

    def test_creation_minimal(self):
        from src.l0.models import L0Entity
        entity = L0Entity(
            mention_text="TP53",
            mention_start=0,
            mention_end=4,
            left_context="",
            right_context=""
        )
        assert entity.mention_text == "TP53"
        assert entity.is_linked is False
        assert entity.linked_entity is None

    def test_creation_with_linked(self):
        from src.l0.models import L0Entity, LinkedEntity
        from src.l2.models import DatabaseRecord

        linked = LinkedEntity(
            entity_id="1",
            label="TP53",
            confidence=0.9,
            start=0,
            end=4,
            matched_text="TP53"
        )

        entity = L0Entity(
            mention_text="TP53",
            mention_start=0,
            mention_end=4,
            left_context="The gene ",
            right_context=" is important",
            candidates=[DatabaseRecord(entity_id="1", label="TP53")],
            num_candidates=1,
            linked_entity=linked,
            is_linked=True,
            pipeline_stage="l3_linked"
        )

        assert entity.is_linked is True
        assert entity.linked_entity.entity_id == "1"
        assert entity.pipeline_stage == "l3_linked"

    def test_default_candidates(self):
        from src.l0.models import L0Entity
        entity = L0Entity(
            mention_text="TP53",
            mention_start=0,
            mention_end=4,
            left_context="",
            right_context=""
        )
        assert entity.candidates == []
        assert entity.num_candidates == 0


class TestL0Output:
    """Tests for L0Output."""

    def test_import(self):
        from src.l0.models import L0Output
        assert L0Output is not None

    def test_creation(self):
        from src.l0.models import L0Output, L0Entity
        entity = L0Entity(
            mention_text="TP53",
            mention_start=0,
            mention_end=4,
            left_context="",
            right_context=""
        )
        output = L0Output(entities=[[entity]], stats={})
        assert len(output.entities) == 1

    def test_with_stats(self):
        from src.l0.models import L0Output
        stats = {
            "total_mentions": 5,
            "linked": 3,
            "unlinked": 2,
            "linking_rate": 0.6
        }
        output = L0Output(entities=[], stats=stats)
        assert output.stats["linking_rate"] == 0.6

    def test_default_stats(self):
        from src.l0.models import L0Output
        output = L0Output(entities=[])
        assert output.stats == {}
