"""
Tests for src/l1/models.py - L1 data models.
"""

import pytest
from pydantic import ValidationError


class TestL1Config:
    """Tests for L1Config."""

    def test_import(self):
        from src.l1.models import L1Config
        assert L1Config is not None

    def test_default_values(self):
        from src.l1.models import L1Config
        config = L1Config()
        assert config.model == "en_core_sci_sm"
        assert config.device == "cpu"
        assert config.batch_size == 16
        assert config.min_entity_length == 2

    def test_custom_values(self):
        from src.l1.models import L1Config
        config = L1Config(
            model="en_core_web_sm",
            device="cuda",
            batch_size=16,
            min_entity_length=3,
            max_left_context=100,
            max_right_context=100
        )
        assert config.model == "en_core_web_sm"
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.min_entity_length == 3

    def test_include_noun_chunks(self):
        from src.l1.models import L1Config
        config = L1Config(include_noun_chunks=True)
        assert config.include_noun_chunks is True


class TestL1Entity:
    """Tests for L1Entity."""

    def test_import(self):
        from src.l1.models import L1Entity
        assert L1Entity is not None

    def test_creation(self):
        from src.l1.models import L1Entity
        entity = L1Entity(
            text="TP53",
            start=0,
            end=4,
            left_context="The ",
            right_context=" gene"
        )
        assert entity.text == "TP53"
        assert entity.start == 0
        assert entity.end == 4

    def test_required_fields(self):
        from src.l1.models import L1Entity
        with pytest.raises(ValidationError):
            L1Entity(text="test")  # Missing other required fields

    def test_context_fields(self):
        from src.l1.models import L1Entity
        entity = L1Entity(
            text="test",
            start=10,
            end=14,
            left_context="prefix ",
            right_context=" suffix"
        )
        assert entity.left_context == "prefix "
        assert entity.right_context == " suffix"


class TestL1Input:
    """Tests for L1Input."""

    def test_import(self):
        from src.l1.models import L1Input
        assert L1Input is not None

    def test_creation(self):
        from src.l1.models import L1Input
        input_data = L1Input(texts=["text1", "text2"])
        assert input_data.texts == ["text1", "text2"]

    def test_empty_texts(self):
        from src.l1.models import L1Input
        input_data = L1Input(texts=[])
        assert input_data.texts == []


class TestL1Output:
    """Tests for L1Output."""

    def test_import(self):
        from src.l1.models import L1Output
        assert L1Output is not None

    def test_creation(self):
        from src.l1.models import L1Output, L1Entity
        entity = L1Entity(
            text="test",
            start=0,
            end=4,
            left_context="",
            right_context=""
        )
        output = L1Output(entities=[[entity]])
        assert len(output.entities) == 1
        assert len(output.entities[0]) == 1

    def test_empty_output(self):
        from src.l1.models import L1Output
        output = L1Output(entities=[])
        assert output.entities == []
