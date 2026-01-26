"""
Tests for src/l2/models.py - L2 data models.
"""

import pytest
from pydantic import ValidationError


class TestDatabaseRecord:
    """Tests for DatabaseRecord."""

    def test_import(self):
        from glinker.l2.models import DatabaseRecord
        assert DatabaseRecord is not None

    def test_creation(self):
        from glinker.l2.models import DatabaseRecord
        record = DatabaseRecord(
            entity_id="MESH:123",
            label="TP53",
            description="A tumor suppressor gene",
            entity_type="gene",
            popularity=1000,
            aliases=["p53", "TRP53"]
        )
        assert record.entity_id == "MESH:123"
        assert record.label == "TP53"

    def test_default_values(self):
        from glinker.l2.models import DatabaseRecord
        record = DatabaseRecord(entity_id="test", label="Test")
        assert record.description == ""
        assert record.entity_type == ""
        assert record.popularity == 0
        assert record.aliases == []

    def test_embedding_fields(self):
        from glinker.l2.models import DatabaseRecord
        record = DatabaseRecord(
            entity_id="test",
            label="Test",
            embedding=[0.1] * 768,
            embedding_model_id="test-model"
        )
        assert record.embedding is not None
        assert len(record.embedding) == 768
        assert record.embedding_model_id == "test-model"

    def test_embedding_default_none(self):
        from glinker.l2.models import DatabaseRecord
        record = DatabaseRecord(entity_id="test", label="Test")
        assert record.embedding is None
        assert record.embedding_model_id is None


class TestLayerConfig:
    """Tests for LayerConfig."""

    def test_import(self):
        from glinker.l2.models import LayerConfig
        assert LayerConfig is not None

    def test_dict_layer_config(self):
        from glinker.l2.models import LayerConfig
        config = LayerConfig(
            type="dict",
            priority=0,
            search_mode=["exact", "fuzzy"]
        )
        assert config.type == "dict"
        assert config.priority == 0

    def test_default_values(self):
        from glinker.l2.models import LayerConfig
        config = LayerConfig(type="dict", priority=0)
        assert config.write is True
        assert config.ttl == 3600
        assert config.cache_policy == "always"

    def test_fuzzy_config(self):
        from glinker.l2.models import LayerConfig, FuzzyConfig
        config = LayerConfig(
            type="dict",
            priority=0,
            fuzzy=FuzzyConfig(min_similarity=0.6, n_gram_size=3)
        )
        assert config.fuzzy.min_similarity == 0.6
        assert config.fuzzy.n_gram_size == 3


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_import(self):
        from glinker.l2.models import EmbeddingConfig
        assert EmbeddingConfig is not None

    def test_creation(self):
        from glinker.l2.models import EmbeddingConfig
        config = EmbeddingConfig(
            enabled=True,
            model_name="test-model",
            dim=768
        )
        assert config.enabled is True
        assert config.dim == 768

    def test_default_values(self):
        from glinker.l2.models import EmbeddingConfig
        config = EmbeddingConfig()
        assert config.enabled is False
        assert config.dim == 768


class TestL2Config:
    """Tests for L2Config."""

    def test_import(self):
        from glinker.l2.models import L2Config
        assert L2Config is not None

    def test_creation(self, l2_config_dict):
        from glinker.l2.models import L2Config
        config = L2Config(**l2_config_dict)
        assert config.max_candidates == 5

    def test_layers_list(self, l2_config_dict):
        from glinker.l2.models import L2Config
        config = L2Config(**l2_config_dict)
        assert len(config.layers) >= 1

    def test_embeddings_config(self, l2_config_dict):
        from glinker.l2.models import L2Config
        config = L2Config(**l2_config_dict)
        assert config.embeddings is not None
        assert config.embeddings.enabled is True


class TestL2Output:
    """Tests for L2Output."""

    def test_import(self):
        from glinker.l2.models import L2Output
        assert L2Output is not None

    def test_creation(self):
        from glinker.l2.models import L2Output, DatabaseRecord
        record = DatabaseRecord(entity_id="test", label="Test")
        output = L2Output(candidates=[[record]])
        assert len(output.candidates) == 1
