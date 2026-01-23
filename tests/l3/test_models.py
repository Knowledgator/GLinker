"""
Tests for src/l3/models.py - L3 data models.
"""

import pytest
from pydantic import ValidationError


class TestL3Config:
    """Tests for L3Config."""

    def test_import(self):
        from src.l3.models import L3Config
        assert L3Config is not None

    def test_creation(self, l3_config_dict):
        from src.l3.models import L3Config
        config = L3Config(**l3_config_dict)
        assert config.model_name == l3_config_dict["model_name"]
        assert config.threshold == 0.3

    def test_default_values(self):
        from src.l3.models import L3Config
        config = L3Config(model_name="test-model")
        assert config.device == "cpu"
        assert config.threshold == 0.5
        assert config.flat_ner is True
        assert config.multi_label is False
        assert config.batch_size == 8

    def test_embedding_settings(self):
        from src.l3.models import L3Config
        config = L3Config(
            model_name="test-model",
            use_precomputed_embeddings=True,
            cache_embeddings=True
        )
        assert config.use_precomputed_embeddings is True
        assert config.cache_embeddings is True

    def test_embedding_settings_default(self):
        from src.l3.models import L3Config
        config = L3Config(model_name="test-model")
        assert config.use_precomputed_embeddings is True
        assert config.cache_embeddings is False


class TestL3Input:
    """Tests for L3Input."""

    def test_import(self):
        from src.l3.models import L3Input
        assert L3Input is not None

    def test_creation(self):
        from src.l3.models import L3Input
        input_data = L3Input(
            texts=["Test text"],
            labels=[["label1", "label2"]]
        )
        assert len(input_data.texts) == 1
        assert len(input_data.labels) == 1

    def test_requires_texts(self):
        from src.l3.models import L3Input
        with pytest.raises(ValidationError):
            L3Input(labels=[["label1"]])

    def test_requires_labels(self):
        from src.l3.models import L3Input
        with pytest.raises(ValidationError):
            L3Input(texts=["text"])


class TestL3Entity:
    """Tests for L3Entity."""

    def test_import(self):
        from src.l3.models import L3Entity
        assert L3Entity is not None

    def test_creation(self):
        from src.l3.models import L3Entity
        entity = L3Entity(
            text="TP53",
            label="Gene",
            start=0,
            end=4,
            score=0.95
        )
        assert entity.text == "TP53"
        assert entity.label == "Gene"
        assert entity.start == 0
        assert entity.end == 4
        assert entity.score == 0.95

    def test_all_fields_required(self):
        from src.l3.models import L3Entity
        with pytest.raises(ValidationError):
            L3Entity(text="TP53", label="Gene")  # Missing start, end, score


class TestL3Output:
    """Tests for L3Output."""

    def test_import(self):
        from src.l3.models import L3Output
        assert L3Output is not None

    def test_creation(self):
        from src.l3.models import L3Output, L3Entity
        entity = L3Entity(text="TP53", label="Gene", start=0, end=4, score=0.9)
        output = L3Output(entities=[[entity]])
        assert len(output.entities) == 1
        assert len(output.entities[0]) == 1

    def test_empty_entities(self):
        from src.l3.models import L3Output
        output = L3Output(entities=[])
        assert output.entities == []

    def test_nested_structure(self):
        from src.l3.models import L3Output, L3Entity
        entity1 = L3Entity(text="TP53", label="Gene", start=0, end=4, score=0.9)
        entity2 = L3Entity(text="BRCA1", label="Gene", start=10, end=15, score=0.85)
        output = L3Output(entities=[[entity1], [entity2]])
        assert len(output.entities) == 2
