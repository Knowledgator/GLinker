"""
Tests for src/l4/processor.py - L4 GLiNER reranking processor.
"""

import pytest


class TestL4ProcessorCreation:
    """Tests for L4Processor initialization."""

    def test_import(self):
        from glinker.l4.processor import L4Processor
        assert L4Processor is not None

    def test_create_via_factory(self, l4_config_dict):
        from glinker.l4.processor import create_l4_processor
        processor = create_l4_processor(l4_config_dict)
        assert processor is not None

    def test_processor_has_component(self, l4_processor):
        assert l4_processor.component is not None

    def test_processor_has_config(self, l4_processor):
        assert l4_processor.config is not None

    def test_processor_has_default_pipeline(self, l4_processor):
        assert l4_processor.pipeline is not None
        assert len(l4_processor.pipeline) > 0


class TestL4ProcessorCall:
    """Tests for L4Processor __call__ method."""

    def test_call_single_text(self, l4_processor, single_text):
        texts = [single_text]
        candidates = [["gene", "disease", "protein"]]
        result = l4_processor(texts=texts, candidates=candidates)

        from glinker.l3.models import L3Output
        assert isinstance(result, L3Output)
        assert len(result.entities) == 1

    def test_call_multiple_texts(self, l4_processor, sample_texts):
        candidates = [
            ["gene", "disease"],
            ["gene", "disease"],
            ["gene", "disease", "drug"]
        ]
        result = l4_processor(texts=sample_texts, candidates=candidates)

        assert len(result.entities) == len(sample_texts)

    def test_call_empty_input(self, l4_processor):
        result = l4_processor(texts=[], candidates=[])
        assert len(result.entities) == 0

    def test_call_empty_text(self, l4_processor):
        result = l4_processor(texts=[""], candidates=[[]])
        assert len(result.entities) == 1
        assert result.entities[0] == []

    def test_result_entities_are_lists(self, l4_processor, single_text):
        result = l4_processor(
            texts=[single_text],
            candidates=[["gene", "disease"]]
        )
        for entities in result.entities:
            assert isinstance(entities, list)

    def test_call_with_input_data(self, l4_processor):
        """Test calling with L3Input object."""
        from glinker.l3.models import L3Input

        input_data = L3Input(
            texts=["TP53 is a gene."],
            labels=[["gene", "disease"]]
        )
        result = l4_processor(input_data=input_data)

        from glinker.l3.models import L3Output
        assert isinstance(result, L3Output)

    def test_call_raises_without_params(self, l4_processor):
        """Should raise ValueError if neither texts+candidates nor input_data provided."""
        with pytest.raises(ValueError):
            l4_processor()


class TestL4ProcessorChunking:
    """Tests for candidate chunking functionality."""

    def test_chunking_with_many_candidates(self, l4_processor):
        """Test that processor handles many candidates via chunking."""
        text = "TP53 is a gene."
        # Create many candidates to force chunking
        many_candidates = [f"label_{i}" for i in range(50)]

        result = l4_processor(
            texts=[text],
            candidates=[many_candidates]
        )

        assert isinstance(result.entities, list)
        assert len(result.entities) == 1

    def test_max_labels_config_used(self, l4_config_dict):
        """Test that max_labels from config is used."""
        from glinker.l4.processor import create_l4_processor

        l4_config_dict["max_labels"] = 5
        processor = create_l4_processor(l4_config_dict)

        assert processor.config.max_labels == 5


class TestL4ProcessorWithL1Entities:
    """Tests for using L1 entities as input_spans."""

    def test_call_with_l1_entities(self, l4_processor, single_text):
        """Test providing L1 entities to constrain predictions."""
        from glinker.l1.models import L1Entity

        # Create mock L1 entities
        l1_entities = [[
            L1Entity(
                text="TP53",
                label="",
                start=0,
                end=4,
                left_context="",
                right_context=" mutations cause breast cancer."
            )
        ]]

        result = l4_processor(
            texts=[single_text],
            candidates=[["gene", "disease"]],
            l1_entities=l1_entities
        )

        assert isinstance(result.entities, list)

    def test_build_input_spans(self, l4_processor):
        """Test _build_input_spans static method."""
        from glinker.l1.models import L1Entity

        l1_entities = [
            L1Entity(text="TP53", label="", start=0, end=4, left_context="", right_context=""),
            L1Entity(text="BRCA1", label="", start=10, end=15, left_context="", right_context="")
        ]

        spans = l4_processor._build_input_spans(l1_entities)

        assert isinstance(spans, list)
        assert len(spans) == 1  # Returns [spans]
        assert len(spans[0]) == 2
        assert spans[0][0] == {"start": 0, "end": 4}
        assert spans[0][1] == {"start": 10, "end": 15}


class TestL4ProcessorSharedCandidates:
    """Tests for shared candidate optimization."""

    def test_shared_candidates_detected(self, l4_processor, sample_texts):
        """Test that shared candidates (same list) are optimized."""
        shared_candidates = ["gene", "disease", "protein"]
        candidates = [shared_candidates] * len(sample_texts)

        result = l4_processor(texts=sample_texts, candidates=candidates)

        assert len(result.entities) == len(sample_texts)

    def test_different_candidates_per_text(self, l4_processor):
        """Test with different candidates per text."""
        texts = ["TP53 is a gene.", "Aspirin is a drug."]
        candidates = [
            ["gene", "disease"],
            ["drug", "chemical"]
        ]

        result = l4_processor(texts=texts, candidates=candidates)

        assert len(result.entities) == 2


class TestL4ProcessorWithSchema:
    """Tests for schema template functionality."""

    def test_processor_with_schema(self, l4_config_dict):
        """Test using schema template for candidates."""
        from glinker.l4.processor import create_l4_processor

        processor = create_l4_processor(l4_config_dict)
        processor.schema = {"template": "{label} - {description}"}

        # Mock candidates with attributes
        class MockCandidate:
            def __init__(self, label, description):
                self.label = label
                self.description = description

            def model_dump(self):
                return {"label": self.label, "description": self.description}

        candidates = [
            MockCandidate("gene", "genetic element"),
            MockCandidate("disease", "medical condition")
        ]

        labels, mapping = processor._create_gliner_labels_with_mapping(candidates)

        assert len(labels) == 2
        assert "gene - genetic element" in labels
        assert "disease - medical condition" in labels

    def test_extract_label_from_object(self, l4_processor):
        """Test _extract_label with object having label attribute."""
        class MockCandidate:
            def __init__(self, label):
                self.label = label

        candidate = MockCandidate("test_label")
        label = l4_processor._extract_label(candidate)

        assert label == "test_label"

    def test_extract_label_from_string(self, l4_processor):
        """Test _extract_label with string candidate."""
        label = l4_processor._extract_label("string_label")
        assert label == "string_label"

    def test_create_labels_with_pydantic(self, l4_processor):
        """Test label creation with Pydantic model."""
        from pydantic import BaseModel

        class Candidate(BaseModel):
            label: str
            description: str

        processor = l4_processor
        processor.schema = {"template": "{label}"}

        candidates = [
            Candidate(label="gene", description="test"),
            Candidate(label="disease", description="test2")
        ]

        labels, mapping = processor._create_gliner_labels_with_mapping(candidates)

        assert len(labels) == 2
        assert "gene" in labels

    def test_create_labels_deduplication(self, l4_processor):
        """Test that duplicate labels are removed."""
        processor = l4_processor
        processor.schema = {"template": "{label}"}

        class Candidate:
            def __init__(self, label):
                self.label = label

            def model_dump(self):
                return {"label": self.label}

        candidates = [
            Candidate("gene"),
            Candidate("Gene"),  # Different case
            Candidate("disease")
        ]

        labels, mapping = processor._create_gliner_labels_with_mapping(candidates)

        # Should deduplicate case-insensitive
        assert len(labels) <= 2


class TestL4ProcessorPipeline:
    """Tests for pipeline execution."""

    def test_custom_pipeline(self, l4_component, l4_config):
        """Test processor with custom pipeline."""
        from glinker.l4.processor import L4Processor

        custom_pipeline = [
            ("predict_entities_chunked", {}),
            ("filter_by_score", {"threshold": 0.7}),
        ]

        processor = L4Processor(l4_config, l4_component, custom_pipeline)
        assert processor.pipeline == custom_pipeline

    def test_default_pipeline(self, l4_processor):
        """Test default pipeline is set correctly."""
        pipeline_methods = [step[0] for step in l4_processor.pipeline]

        assert "predict_entities_chunked" in pipeline_methods
        assert "deduplicate_entities" in pipeline_methods
        assert "filter_by_score" in pipeline_methods
        assert "sort_by_position" in pipeline_methods


class TestL4ProcessorModels:
    """Tests for L4 models."""

    def test_l4_config_import(self):
        from glinker.l4.models import L4Config
        assert L4Config is not None

    def test_l4_config_creation(self, l4_config_dict):
        from glinker.l4.models import L4Config
        config = L4Config(**l4_config_dict)

        assert config.model_name == l4_config_dict["model_name"]
        assert config.threshold == l4_config_dict["threshold"]
        assert config.max_labels == l4_config_dict.get("max_labels", 20)

    def test_l4_config_defaults(self):
        from glinker.l4.models import L4Config
        config = L4Config(model_name="test-model")

        assert config.device == "cpu"
        assert config.threshold == 0.5
        assert config.max_labels == 20
        assert config.flat_ner is True
        assert config.multi_label is False
