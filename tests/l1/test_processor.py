"""
Tests for src/l1/processor.py - L1 NER processor.
"""

import pytest


def create_processor(processor_name, config_dict):
    """Helper to create processor from registry."""
    from glinker.core.registry import processor_registry
    factory = processor_registry.get(processor_name)
    return factory(config_dict=config_dict, pipeline=None)


class TestL1ProcessorCreation:
    """Tests for L1 processor initialization."""

    def test_create_via_registry(self, l1_config_dict):
        processor = create_processor("l1_batch", l1_config_dict)
        assert processor is not None

    def test_processor_has_component(self, l1_config_dict):
        processor = create_processor("l1_batch", l1_config_dict)
        assert hasattr(processor, 'component')
        assert processor.component is not None

    def test_processor_has_config(self, l1_config_dict):
        processor = create_processor("l1_batch", l1_config_dict)
        assert hasattr(processor, 'config')


class TestL1ProcessorCall:
    """Tests for L1 processor __call__ method."""

    def test_call_single_text(self, l1_config_dict):
        from glinker.l1.models import L1Output

        processor = create_processor("l1_batch", l1_config_dict)
        result = processor(texts=["TP53 causes cancer."])

        assert isinstance(result, L1Output)
        assert len(result.entities) == 1

    def test_call_multiple_texts(self, l1_config_dict):
        processor = create_processor("l1_batch", l1_config_dict)
        texts = ["TP53 is a gene.", "BRCA1 causes cancer."]
        result = processor(texts=texts)

        assert len(result.entities) == 2

    def test_call_empty_input(self, l1_config_dict):
        processor = create_processor("l1_batch", l1_config_dict)
        result = processor(texts=[])

        assert result.entities == []

    def test_call_empty_text(self, l1_config_dict):
        processor = create_processor("l1_batch", l1_config_dict)
        result = processor(texts=[""])

        assert len(result.entities) == 1
        assert result.entities[0] == []

    def test_result_entities_are_lists(self, l1_config_dict):
        processor = create_processor("l1_batch", l1_config_dict)
        result = processor(texts=["TP53 test"])

        for text_entities in result.entities:
            assert isinstance(text_entities, list)


class TestL1ProcessorBatching:
    """Tests for L1 processor batch processing."""

    def test_batch_size_1(self, l1_config_dict):
        config = {**l1_config_dict, "batch_size": 1}
        processor = create_processor("l1_batch", config)

        texts = ["Text 1.", "Text 2.", "Text 3."]
        result = processor(texts=texts)

        assert len(result.entities) == 3

    def test_large_batch(self, l1_config_dict):
        config = {**l1_config_dict, "batch_size": 100}
        processor = create_processor("l1_batch", config)

        texts = [f"Text {i}." for i in range(10)]
        result = processor(texts=texts)

        assert len(result.entities) == 10
