"""
Tests for src/l3/processor.py - L3 GLiNER processor.
"""

import pytest


def create_processor(processor_name, config_dict):
    """Helper to create processor from registry."""
    from glinker.core.registry import processor_registry
    factory = processor_registry.get(processor_name)
    return factory(config_dict=config_dict, pipeline=None)


class TestL3ProcessorCreation:
    """Tests for L3 processor initialization."""

    def test_create_via_registry(self, l3_config_dict):
        processor = create_processor("l3_batch", l3_config_dict)
        assert processor is not None

    def test_processor_has_component(self, l3_config_dict):
        processor = create_processor("l3_batch", l3_config_dict)
        assert hasattr(processor, 'component')
        assert processor.component is not None

    def test_processor_has_config(self, l3_config_dict):
        processor = create_processor("l3_batch", l3_config_dict)
        assert hasattr(processor, 'config')

    def test_processor_has_schema(self, l3_config_dict):
        processor = create_processor("l3_batch", l3_config_dict)
        assert hasattr(processor, 'schema')


class TestL3ProcessorCall:
    """Tests for L3 processor __call__ method."""

    def test_call_with_texts_and_candidates(self, l3_config_dict):
        from glinker.l3.models import L3Output
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l3_batch", l3_config_dict)

        # Create candidates
        candidates = [
            DatabaseRecord(entity_id="1", label="gene", description="A gene"),
            DatabaseRecord(entity_id="2", label="disease", description="A disease"),
        ]

        result = processor(
            texts=["TP53 causes cancer."],
            candidates=[[candidates[0], candidates[1]]]
        )

        assert isinstance(result, L3Output)
        assert len(result.entities) == 1

    def test_call_empty_texts(self, l3_config_dict):
        from glinker.l3.models import L3Output

        processor = create_processor("l3_batch", l3_config_dict)
        result = processor(texts=[], candidates=[])

        assert isinstance(result, L3Output)
        assert result.entities == []

    def test_call_empty_candidates(self, l3_config_dict):
        from glinker.l3.models import L3Output

        processor = create_processor("l3_batch", l3_config_dict)
        result = processor(texts=["Some text"], candidates=[[]])

        assert isinstance(result, L3Output)
        assert len(result.entities) == 1


class TestL3ProcessorLabelCreation:
    """Tests for label creation methods."""

    def test_extract_label_from_record(self, l3_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l3_batch", l3_config_dict)
        record = DatabaseRecord(entity_id="1", label="TP53", description="A gene")

        label = processor._extract_label(record)
        assert label == "TP53"

    def test_extract_label_from_string(self, l3_config_dict):
        processor = create_processor("l3_batch", l3_config_dict)
        label = processor._extract_label("simple_label")
        assert label == "simple_label"

    def test_create_gliner_labels_with_template(self, l3_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l3_batch", l3_config_dict)
        processor.schema = {"template": "{label}: {description}"}

        candidates = [
            DatabaseRecord(entity_id="1", label="TP53", description="Tumor protein"),
            DatabaseRecord(entity_id="2", label="BRCA1", description="Breast cancer gene"),
        ]

        labels, mapping = processor._create_gliner_labels_with_mapping(candidates)

        assert len(labels) == 2
        assert "TP53: Tumor protein" in labels
        assert "BRCA1: Breast cancer gene" in labels
        assert len(mapping) == 2

    def test_create_gliner_labels_deduplication(self, l3_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l3_batch", l3_config_dict)
        processor.schema = {"template": "{label}"}

        candidates = [
            DatabaseRecord(entity_id="1", label="TP53"),
            DatabaseRecord(entity_id="2", label="TP53"),  # Duplicate
            DatabaseRecord(entity_id="3", label="tp53"),  # Case-insensitive duplicate
        ]

        labels, mapping = processor._create_gliner_labels_with_mapping(candidates)

        assert len(labels) == 1  # Should deduplicate


class TestL3ProcessorRanking:
    """Tests for entity ranking."""

    def test_rank_entities(self, l3_config_dict):
        from glinker.l2.models import DatabaseRecord
        from glinker.l3.models import L3Entity

        processor = create_processor("l3_batch", l3_config_dict)
        processor.schema = {
            "template": "{label}",
            "ranking": [
                {"field": "gliner_score", "weight": 0.7},
                {"field": "popularity", "weight": 0.3}
            ]
        }

        entities = [
            L3Entity(text="TP53", label="TP53", start=0, end=4, score=0.5),
            L3Entity(text="BRCA1", label="BRCA1", start=10, end=15, score=0.9),
        ]

        candidates = [
            DatabaseRecord(entity_id="1", label="TP53", popularity=1000000),
            DatabaseRecord(entity_id="2", label="BRCA1", popularity=100),
        ]

        ranked = processor._rank_entities(entities, candidates)
        # Should be re-ranked based on combined score
        assert isinstance(ranked, list)
        assert len(ranked) == 2


class TestL3ProcessorPrecomputed:
    """Tests for precomputed embeddings usage."""

    def test_can_use_precomputed_no_embeddings(self, l3_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l3_batch", l3_config_dict)

        candidates = [
            DatabaseRecord(entity_id="1", label="TP53"),  # No embedding
        ]

        result = processor._can_use_precomputed(candidates, {})
        assert result is False

    def test_can_use_precomputed_with_embeddings(self, l3_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l3_batch", l3_config_dict)

        candidates = [
            DatabaseRecord(
                entity_id="1",
                label="TP53",
                embedding=[0.1] * 768,
                embedding_model_id=l3_config_dict["model_name"]
            ),
        ]

        result = processor._can_use_precomputed(candidates, {})
        assert result is True

    def test_can_use_precomputed_model_mismatch(self, l3_config_dict):
        from glinker.l2.models import DatabaseRecord

        processor = create_processor("l3_batch", l3_config_dict)

        candidates = [
            DatabaseRecord(
                entity_id="1",
                label="TP53",
                embedding=[0.1] * 768,
                embedding_model_id="different-model"  # Wrong model
            ),
        ]

        result = processor._can_use_precomputed(candidates, {})
        assert result is False
