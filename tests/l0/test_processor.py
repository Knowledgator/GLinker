"""
Tests for src/l0/processor.py - L0 aggregation processor.
"""

import pytest


def create_processor(processor_name, config_dict):
    """Helper to create processor from registry."""
    from src.core.registry import processor_registry
    factory = processor_registry.get(processor_name)
    return factory(config_dict=config_dict, pipeline=None)


class TestL0ProcessorCreation:
    """Tests for L0 processor initialization."""

    def test_create_via_registry(self, l0_config_dict):
        processor = create_processor("l0_aggregator", l0_config_dict)
        assert processor is not None

    def test_processor_has_component(self, l0_config_dict):
        processor = create_processor("l0_aggregator", l0_config_dict)
        assert hasattr(processor, 'component')
        assert processor.component is not None

    def test_processor_has_config(self, l0_config_dict):
        processor = create_processor("l0_aggregator", l0_config_dict)
        assert hasattr(processor, 'config')

    def test_processor_has_schema(self, l0_config_dict):
        processor = create_processor("l0_aggregator", l0_config_dict)
        assert hasattr(processor, 'schema')

    def test_default_pipeline(self, l0_config_dict):
        processor = create_processor("l0_aggregator", l0_config_dict)
        pipeline = processor._default_pipeline()
        assert ("aggregate", {}) in pipeline
        assert ("calculate_stats", {}) in pipeline


class TestL0ProcessorCall:
    """Tests for L0 processor __call__ method."""

    def test_call_with_all_params(self, l0_config_dict):
        from src.l0.models import L0Output
        from src.l1.models import L1Entity
        from src.l2.models import DatabaseRecord
        from src.l3.models import L3Entity

        processor = create_processor("l0_aggregator", l0_config_dict)

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53", description="Gene")
        ]]
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9)
        ]]

        result = processor(
            l1_entities=l1_entities,
            l2_candidates=l2_candidates,
            l3_entities=l3_entities
        )

        assert isinstance(result, L0Output)
        assert len(result.entities) == 1
        assert "total_mentions" in result.stats

    def test_call_with_input_data(self, l0_config_dict):
        from src.l0.models import L0Output, L0Input
        from src.l1.models import L1Entity
        from src.l2.models import DatabaseRecord
        from src.l3.models import L3Entity

        processor = create_processor("l0_aggregator", l0_config_dict)

        input_data = L0Input(
            l1_entities=[[
                L1Entity(text="TP53", start=0, end=4,
                         left_context="", right_context="")
            ]],
            l2_candidates=[[
                DatabaseRecord(entity_id="1", label="TP53")
            ]],
            l3_entities=[[
                L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9)
            ]]
        )

        result = processor(input_data=input_data)

        assert isinstance(result, L0Output)
        assert len(result.entities) == 1

    def test_call_empty_inputs(self, l0_config_dict):
        from src.l0.models import L0Output

        processor = create_processor("l0_aggregator", l0_config_dict)
        result = processor(l1_entities=[], l2_candidates=[], l3_entities=[])

        assert isinstance(result, L0Output)
        assert result.entities == []
        assert result.stats["total_mentions"] == 0

    def test_call_missing_params_raises(self, l0_config_dict):
        processor = create_processor("l0_aggregator", l0_config_dict)

        with pytest.raises(ValueError):
            processor()  # No params

        with pytest.raises(ValueError):
            processor(l1_entities=[])  # Missing l2 and l3


class TestL0ProcessorWithSchema:
    """Tests for L0 processor with schema template."""

    def test_processor_uses_schema_template(self, l0_config_dict):
        from src.l1.models import L1Entity
        from src.l2.models import DatabaseRecord
        from src.l3.models import L3Entity

        processor = create_processor("l0_aggregator", l0_config_dict)
        processor.schema = {"template": "{label}: {description}"}

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53", description="Tumor protein")
        ]]
        l3_entities = [[
            L3Entity(
                text="TP53",
                label="TP53: Tumor protein",  # Formatted label from L3
                start=0, end=4, score=0.9
            )
        ]]

        result = processor(
            l1_entities=l1_entities,
            l2_candidates=l2_candidates,
            l3_entities=l3_entities
        )

        # Should match using template
        assert result.entities[0][0].is_linked is True


class TestL0ProcessorStats:
    """Tests for L0 processor statistics."""

    def test_stats_in_output(self, l0_config_dict):
        from src.l1.models import L1Entity
        from src.l2.models import DatabaseRecord
        from src.l3.models import L3Entity

        processor = create_processor("l0_aggregator", l0_config_dict)

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context=""),
            L1Entity(text="UNKNOWN", start=10, end=17,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53")
        ]]
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9)
        ]]

        result = processor(
            l1_entities=l1_entities,
            l2_candidates=l2_candidates,
            l3_entities=l3_entities
        )

        assert result.stats["total_mentions"] >= 1
        assert result.stats["linked"] >= 1
        assert "stages" in result.stats


class TestL0ProcessorPipeline:
    """Tests for L0 processor pipeline execution."""

    def test_pipeline_filter_by_confidence(self, l0_config_dict):
        from src.l1.models import L1Entity
        from src.l2.models import DatabaseRecord
        from src.l3.models import L3Entity

        # Set min_confidence to filter low-confidence entities
        config = {**l0_config_dict, "min_confidence": 0.5}
        processor = create_processor("l0_aggregator", config)

        l1_entities = [[
            L1Entity(text="TP53", start=0, end=4,
                     left_context="", right_context=""),
            L1Entity(text="BRCA1", start=10, end=15,
                     left_context="", right_context="")
        ]]
        l2_candidates = [[
            DatabaseRecord(entity_id="1", label="TP53"),
            DatabaseRecord(entity_id="2", label="BRCA1")
        ]]
        l3_entities = [[
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.9),  # High confidence
            L3Entity(text="BRCA1", label="BRCA1", start=10, end=15, score=0.3)  # Low confidence
        ]]

        result = processor(
            l1_entities=l1_entities,
            l2_candidates=l2_candidates,
            l3_entities=l3_entities
        )

        # Only high-confidence entity should pass filter
        linked_entities = [e for e in result.entities[0] if e.is_linked]
        assert len(linked_entities) == 1
        assert linked_entities[0].mention_text == "TP53"
