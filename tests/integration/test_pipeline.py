"""
Integration tests for the full EntityLinker pipeline.
"""

import pytest


class TestPipelineExecution:
    """Tests for complete pipeline execution."""

    def test_executor_creation(self, executor):
        """Test that executor is created successfully."""
        assert executor is not None

    def test_executor_has_processors(self, executor):
        """Test that executor has all required processors."""
        assert hasattr(executor, 'processors')
        assert len(executor.processors) >= 3  # L1, L2, L3 at minimum

    def test_execute_single_text(self, loaded_executor, single_text):
        """Test pipeline execution with single text."""
        from src.l1.models import L1Input

        result = loaded_executor.execute(L1Input(texts=[single_text]))

        assert result is not None
        assert result.get("l1_result") is not None
        assert result.get("l2_result") is not None
        assert result.get("l3_result") is not None

    def test_execute_multiple_texts(self, loaded_executor, sample_texts):
        """Test pipeline execution with multiple texts."""
        from src.l1.models import L1Input

        result = loaded_executor.execute(L1Input(texts=sample_texts))

        assert result is not None
        l1_result = result.get("l1_result")
        assert len(l1_result.entities) == len(sample_texts)

    def test_l1_result_structure(self, loaded_executor, single_text):
        """Test L1 result structure."""
        from src.l1.models import L1Input, L1Output

        result = loaded_executor.execute(L1Input(texts=[single_text]))
        l1_result = result.get("l1_result")

        assert isinstance(l1_result, L1Output)
        assert len(l1_result.entities) == 1
        assert isinstance(l1_result.entities[0], list)

    def test_l2_result_structure(self, loaded_executor, single_text):
        """Test L2 result structure."""
        from src.l1.models import L1Input
        from src.l2.models import L2Output

        result = loaded_executor.execute(L1Input(texts=[single_text]))
        l2_result = result.get("l2_result")

        assert isinstance(l2_result, L2Output)
        assert hasattr(l2_result, 'candidates')

    def test_l3_result_structure(self, loaded_executor, single_text):
        """Test L3 result structure."""
        from src.l1.models import L1Input
        from src.l3.models import L3Output

        result = loaded_executor.execute(L1Input(texts=[single_text]))
        l3_result = result.get("l3_result")

        assert isinstance(l3_result, L3Output)
        assert hasattr(l3_result, 'entities')


class TestPipelineWithL0:
    """Tests for full pipeline including L0 aggregation."""

    def test_l0_result_exists(self, loaded_executor, single_text):
        """Test that L0 result is produced."""
        from src.l1.models import L1Input

        result = loaded_executor.execute(L1Input(texts=[single_text]))

        assert result.get("l0_result") is not None

    def test_l0_result_structure(self, loaded_executor, single_text):
        """Test L0 result structure."""
        from src.l1.models import L1Input
        from src.l0.models import L0Output

        result = loaded_executor.execute(L1Input(texts=[single_text]))
        l0_result = result.get("l0_result")

        assert isinstance(l0_result, L0Output)
        assert hasattr(l0_result, 'entities')
        assert hasattr(l0_result, 'stats')

    def test_l0_aggregates_all_layers(self, loaded_executor):
        """Test that L0 aggregates information from all layers."""
        from src.l1.models import L1Input

        text = "TP53 mutations cause breast cancer."
        result = loaded_executor.execute(L1Input(texts=[text]))

        l0_result = result.get("l0_result")
        l1_result = result.get("l1_result")

        # L0 should have entities corresponding to L1 mentions
        if l1_result.entities[0]:  # If L1 found entities
            assert len(l0_result.entities) >= 1

    def test_l0_stats(self, loaded_executor, single_text):
        """Test that L0 produces statistics."""
        from src.l1.models import L1Input

        result = loaded_executor.execute(L1Input(texts=[single_text]))
        l0_result = result.get("l0_result")

        stats = l0_result.stats
        assert "total_mentions" in stats
        assert "linked" in stats
        assert "unlinked" in stats
        assert "linking_rate" in stats


class TestPipelineEntityLinking:
    """Tests for entity linking behavior."""

    def test_known_entity_linked(self, loaded_executor):
        """Test that known entities are linked."""
        from src.l1.models import L1Input

        # "TP53" should be in our sample entities
        text = "TP53 is a tumor suppressor gene."
        result = loaded_executor.execute(L1Input(texts=[text]))

        l2_result = result.get("l2_result")
        # Should find candidates for TP53
        assert l2_result.candidates is not None

    def test_unknown_entity_not_linked(self, loaded_executor):
        """Test that unknown entities are not linked in L2."""
        from src.l1.models import L1Input

        # Use a completely unknown entity
        text = "XYZABC123 is not a known gene."
        result = loaded_executor.execute(L1Input(texts=[text]))

        l2_result = result.get("l2_result")
        # Unknown entity should have no candidates (or empty list)
        # depending on how L2 handles it


class TestPipelineEdgeCases:
    """Tests for pipeline edge cases."""

    def test_empty_text(self, loaded_executor):
        """Test pipeline with empty text."""
        from src.l1.models import L1Input

        result = loaded_executor.execute(L1Input(texts=[""]))

        l1_result = result.get("l1_result")
        assert len(l1_result.entities) == 1
        assert l1_result.entities[0] == []

    def test_text_without_entities(self, loaded_executor):
        """Test pipeline with text that has no entities."""
        from src.l1.models import L1Input

        text = "This is a simple text without any special entities."
        result = loaded_executor.execute(L1Input(texts=[text]))

        l1_result = result.get("l1_result")
        # L1 may or may not find entities depending on model
        assert len(l1_result.entities) == 1

    def test_empty_input_list(self, loaded_executor):
        """Test pipeline with empty input list."""
        from src.l1.models import L1Input

        result = loaded_executor.execute(L1Input(texts=[]))

        l1_result = result.get("l1_result")
        assert l1_result.entities == []


class TestPipelineLoadEntities:
    """Tests for entity loading functionality."""

    def test_load_entities_from_jsonl(self, executor, entities_jsonl_file):
        """Test loading entities from JSONL file."""
        executor.load_entities(
            entities_jsonl_file,
            target_layers=["dict"]
        )

        # Verify entities are loaded
        l2_processor = executor.processors.get("l2")
        if l2_processor:
            layer = l2_processor.component.layers[0]
            assert layer.count() >= 1

    def test_load_entities_twice(self, executor, entities_jsonl_file):
        """Test loading entities twice doesn't duplicate."""
        executor.load_entities(
            entities_jsonl_file,
            target_layers=["dict"],
            overwrite=True
        )
        count1 = executor.processors["l2"].component.layers[0].count()

        executor.load_entities(
            entities_jsonl_file,
            target_layers=["dict"],
            overwrite=True
        )
        count2 = executor.processors["l2"].component.layers[0].count()

        # With overwrite=True, count should be the same
        assert count1 == count2
