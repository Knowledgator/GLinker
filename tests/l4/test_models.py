"""
Tests for src/l4/models.py - L4 data models.
"""

import pytest


class TestL4Config:
    """Tests for L4Config model."""

    def test_import(self):
        from glinker.l4.models import L4Config
        assert L4Config is not None

    def test_creation_minimal(self):
        from glinker.l4.models import L4Config
        config = L4Config(model_name="test-model")
        assert config.model_name == "test-model"

    def test_creation_full(self, l4_config_dict):
        from glinker.l4.models import L4Config
        config = L4Config(**l4_config_dict)
        assert config.model_name == l4_config_dict["model_name"]
        assert config.device == l4_config_dict["device"]
        assert config.threshold == l4_config_dict["threshold"]

    def test_defaults(self):
        from glinker.l4.models import L4Config
        config = L4Config(model_name="test")

        assert config.device == "cpu"
        assert config.threshold == 0.5
        assert config.flat_ner is True
        assert config.multi_label is False
        assert config.max_labels == 20
        assert config.token is None
        assert config.max_length is None

    def test_max_labels_default(self):
        from glinker.l4.models import L4Config
        config = L4Config(model_name="test")
        assert config.max_labels == 20

    def test_max_labels_custom(self):
        from glinker.l4.models import L4Config
        config = L4Config(model_name="test", max_labels=50)
        assert config.max_labels == 50

    def test_max_length_optional(self):
        from glinker.l4.models import L4Config
        config = L4Config(model_name="test")
        assert config.max_length is None

        config_with_length = L4Config(model_name="test", max_length=256)
        assert config_with_length.max_length == 256

    def test_token_optional(self):
        from glinker.l4.models import L4Config
        config = L4Config(model_name="test")
        assert config.token is None

        config_with_token = L4Config(model_name="test", token="hf_test")
        assert config_with_token.token == "hf_test"

    def test_config_is_base_config(self):
        from glinker.l4.models import L4Config
        from glinker.core.base import BaseConfig
        config = L4Config(model_name="test")
        assert isinstance(config, BaseConfig)

    def test_field_types(self):
        from glinker.l4.models import L4Config
        config = L4Config(
            model_name="test",
            token="token",
            device="cuda",
            threshold=0.8,
            flat_ner=False,
            multi_label=True,
            max_labels=30,
            max_length=512
        )

        assert isinstance(config.model_name, str)
        assert isinstance(config.token, str)
        assert isinstance(config.device, str)
        assert isinstance(config.threshold, float)
        assert isinstance(config.flat_ner, bool)
        assert isinstance(config.multi_label, bool)
        assert isinstance(config.max_labels, int)
        assert isinstance(config.max_length, int)
