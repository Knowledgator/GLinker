"""
Tests for src/l2/processor.py - L2 database processor.
"""

import pytest


def create_processor(processor_name, config_dict):
    """Helper to create processor from registry."""
    from glinker.core.registry import processor_registry
    factory = processor_registry.get(processor_name)
    return factory(config_dict=config_dict, pipeline=None)


class TestL2ProcessorCreation:
    """Tests for L2 processor initialization."""

    def test_create_via_registry(self, l2_config_dict):
        processor = create_processor("l2_chain", l2_config_dict)
        assert processor is not None

    def test_processor_has_component(self, l2_config_dict):
        processor = create_processor("l2_chain", l2_config_dict)
        assert hasattr(processor, 'component')
        assert processor.component is not None

    def test_processor_has_schema(self, l2_config_dict):
        processor = create_processor("l2_chain", l2_config_dict)
        assert hasattr(processor, 'schema')


class TestL2ProcessorFormatLabel:
    """Tests for format_label method."""

    def test_format_label_simple(self, l2_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l2_chain", l2_config_dict)
        processor.schema = {"template": "{label}"}

        record = DatabaseRecord(
            entity_id="test",
            label="TP53",
            description="Tumor protein"
        )

        formatted = processor.format_label(record)
        assert formatted == "TP53"

    def test_format_label_with_description(self, l2_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l2_chain", l2_config_dict)
        processor.schema = {"template": "{label}: {description}"}

        record = DatabaseRecord(
            entity_id="test",
            label="TP53",
            description="Tumor protein p53"
        )

        formatted = processor.format_label(record)
        assert formatted == "TP53: Tumor protein p53"

    def test_format_label_default_template(self, l2_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l2_chain", l2_config_dict)
        processor.schema = {}  # No template

        record = DatabaseRecord(entity_id="test", label="TP53")

        formatted = processor.format_label(record)
        assert formatted == "TP53"


class TestL2ProcessorCall:
    """Tests for L2 processor __call__ method."""

    def test_call_with_mentions(self, l2_config_dict, sample_entities):
        from glinker.l2.models import L2Output, DatabaseRecord

        processor = create_processor("l2_chain", l2_config_dict)

        # Load entities
        records = [
            DatabaseRecord(
                entity_id=e["entity_id"],
                label=e["label"],
                description=e["description"],
                entity_type=e["entity_type"],
                popularity=e["popularity"],
                aliases=e["aliases"]
            )
            for e in sample_entities
        ]
        processor.component.layers[0].load_bulk(records)

        # Call with mentions
        from glinker.l1.models import L1Entity
        mentions = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]

        result = processor(mentions=mentions)

        assert isinstance(result, L2Output)
        assert len(result.candidates) == 1

    def test_call_empty_mentions(self, l2_config_dict):
        from glinker.l2.models import L2Output

        processor = create_processor("l2_chain", l2_config_dict)
        result = processor(mentions=[])

        assert isinstance(result, L2Output)
        # Result is empty nested list for consistency with input structure
        assert result.candidates == [] or result.candidates == [[]]


class TestL2ProcessorPrecompute:
    """Tests for L2 processor precompute_embeddings."""

    def test_precompute_method_exists(self, l2_config_dict):
        processor = create_processor("l2_chain", l2_config_dict)
        assert hasattr(processor, 'precompute_embeddings')
