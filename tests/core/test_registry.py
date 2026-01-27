"""
Tests for src/core/registry.py - Processor registry.
"""

import pytest


class TestProcessorRegistry:
    """Tests for ProcessorRegistry."""

    def test_registry_import(self):
        from glinker.core.registry import processor_registry
        assert processor_registry is not None

    def test_registry_has_processors(self):
        from glinker.core.registry import processor_registry
        assert len(processor_registry._registry) > 0

    def test_l1_spacy_registered(self):
        from glinker.core.registry import processor_registry
        assert "l1_spacy" in processor_registry._registry

    def test_l2_chain_registered(self):
        from glinker.core.registry import processor_registry
        assert "l2_chain" in processor_registry._registry

    def test_l3_batch_registered(self):
        from glinker.core.registry import processor_registry
        assert "l3_batch" in processor_registry._registry

    def test_l0_aggregator_registered(self):
        from glinker.core.registry import processor_registry
        assert "l0_aggregator" in processor_registry._registry

    def test_create_l1_processor(self, l1_config_dict):
        from glinker.core.registry import processor_registry
        factory = processor_registry.get("l1_spacy")
        processor = factory(config_dict=l1_config_dict, pipeline=None)
        assert processor is not None

    def test_create_l2_processor(self, l2_config_dict):
        from glinker.core.registry import processor_registry
        factory = processor_registry.get("l2_chain")
        processor = factory(config_dict=l2_config_dict, pipeline=None)
        assert processor is not None

    def test_create_l0_processor(self, l0_config_dict):
        from glinker.core.registry import processor_registry
        factory = processor_registry.get("l0_aggregator")
        processor = factory(config_dict=l0_config_dict, pipeline=None)
        assert processor is not None

    def test_create_unknown_processor_raises(self):
        from glinker.core.registry import processor_registry
        with pytest.raises(KeyError):
            processor_registry.get("nonexistent_processor")

    def test_register_decorator(self):
        from glinker.core.registry import processor_registry

        @processor_registry.register("test_processor_temp")
        def create_test_processor(config_dict, pipeline):
            return "test"

        assert "test_processor_temp" in processor_registry._registry

        # Cleanup
        del processor_registry._registry["test_processor_temp"]

    def test_list_available(self):
        from glinker.core.registry import processor_registry
        available = processor_registry.list_available()
        assert isinstance(available, list)
        assert "l1_spacy" in available
        assert "l2_chain" in available
