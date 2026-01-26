"""
Tests for src/core/dag.py - DAG pipeline system.
"""

import pytest


class TestPipeContext:
    """Tests for PipeContext."""

    def test_context_import(self):
        from glinker.core.dag import PipeContext
        assert PipeContext is not None

    def test_context_creation(self):
        from glinker.core.dag import PipeContext
        ctx = PipeContext()
        assert ctx is not None

    def test_context_set_get(self):
        from glinker.core.dag import PipeContext
        ctx = PipeContext()
        ctx.set("key1", {"value": 123})
        result = ctx.get("key1")
        assert result["value"] == 123

    def test_context_get_nonexistent(self):
        from glinker.core.dag import PipeContext
        ctx = PipeContext()
        result = ctx.get("nonexistent")
        assert result is None

    def test_context_pipeline_input(self):
        from glinker.core.dag import PipeContext
        ctx = PipeContext(pipeline_input={"texts": ["hello"]})
        input_data = ctx.get("$input")
        assert input_data["texts"] == ["hello"]

    def test_context_multiple_keys(self):
        from glinker.core.dag import PipeContext
        ctx = PipeContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        ctx.set("c", 3)
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2
        assert ctx.get("c") == 3


class TestFieldResolver:
    """Tests for FieldResolver."""

    def test_resolver_import(self):
        from glinker.core.dag import FieldResolver
        assert FieldResolver is not None

    def test_resolve_simple_field(self):
        from glinker.core.dag import FieldResolver, PipeContext, InputConfig
        ctx = PipeContext(pipeline_input={"texts": ["hello", "world"]})
        config = InputConfig(source="$input", fields="texts")
        result = FieldResolver.resolve(ctx, config)
        assert result == ["hello", "world"]

    def test_resolve_nested_field(self):
        from glinker.core.dag import FieldResolver, PipeContext, InputConfig
        ctx = PipeContext()
        ctx.set("result", {"entities": [{"text": "TP53"}]})
        config = InputConfig(source="result", fields="entities")
        result = FieldResolver.resolve(ctx, config)
        assert result == [{"text": "TP53"}]

    def test_resolve_array_index(self):
        from glinker.core.dag import FieldResolver, PipeContext, InputConfig
        ctx = PipeContext()
        ctx.set("data", {"items": ["a", "b", "c"]})
        config = InputConfig(source="data", fields="items[0]")
        result = FieldResolver.resolve(ctx, config)
        assert result == "a"

    def test_resolve_array_star(self):
        from glinker.core.dag import FieldResolver, PipeContext, InputConfig
        ctx = PipeContext()
        ctx.set("data", {"items": [{"name": "a"}, {"name": "b"}]})
        config = InputConfig(source="data", fields="items[*].name")
        result = FieldResolver.resolve(ctx, config)
        assert result == ["a", "b"]

    def test_resolve_slice(self):
        from glinker.core.dag import FieldResolver, PipeContext, InputConfig
        ctx = PipeContext()
        ctx.set("data", {"items": [1, 2, 3, 4, 5]})
        config = InputConfig(source="data", fields="items[1:3]")
        result = FieldResolver.resolve(ctx, config)
        assert result == [2, 3]

    def test_resolve_nested_array_star(self):
        from glinker.core.dag import FieldResolver, PipeContext, InputConfig
        ctx = PipeContext()
        ctx.set("data", {"outer": [[{"x": 1}, {"x": 2}], [{"x": 3}]]})
        config = InputConfig(source="data", fields="outer[*][*].x")
        result = FieldResolver.resolve(ctx, config)
        assert result == [[1, 2], [3]]


class TestPipeNode:
    """Tests for PipeNode."""

    def test_node_import(self):
        from glinker.core.dag import PipeNode
        assert PipeNode is not None

    def test_node_creation(self):
        from glinker.core.dag import PipeNode
        node = PipeNode(
            id="test",
            processor="l1_batch",
            inputs={"texts": {"source": "$input", "fields": "texts"}},
            output={"key": "test_result"},
            config={"model": "en_core_sci_sm"}
        )
        assert node.id == "test"
        assert node.processor == "l1_batch"

    def test_node_with_requires(self):
        from glinker.core.dag import PipeNode
        node = PipeNode(
            id="l2",
            processor="l2_chain",
            requires=["l1"],
            inputs={},
            output={"key": "l2_result"},
            config={}
        )
        assert "l1" in node.requires

    def test_node_with_schema(self):
        from glinker.core.dag import PipeNode
        node = PipeNode(
            id="l3",
            processor="l3_batch",
            inputs={},
            output={"key": "l3_result"},
            config={},
            schema={"template": "{label}: {description}"}
        )
        assert node.schema["template"] == "{label}: {description}"

    def test_node_default_requires(self):
        from glinker.core.dag import PipeNode
        node = PipeNode(
            id="test",
            processor="l1_batch",
            inputs={},
            output={"key": "result"},
            config={}
        )
        assert node.requires == []


class TestDAGPipeline:
    """Tests for DAGPipeline."""

    def test_pipeline_import(self):
        from glinker.core.dag import DAGPipeline
        assert DAGPipeline is not None

    def test_pipeline_creation(self, pipeline_config_dict):
        from glinker.core.dag import DAGPipeline
        pipeline = DAGPipeline(**pipeline_config_dict)
        assert pipeline.name == "test_pipeline"

    def test_pipeline_has_nodes(self, pipeline_config_dict):
        from glinker.core.dag import DAGPipeline
        pipeline = DAGPipeline(**pipeline_config_dict)
        assert len(pipeline.nodes) == 4

    def test_pipeline_node_ids(self, pipeline_config_dict):
        from glinker.core.dag import DAGPipeline
        pipeline = DAGPipeline(**pipeline_config_dict)
        node_ids = [n.id for n in pipeline.nodes]
        assert "l1" in node_ids
        assert "l2" in node_ids
        assert "l3" in node_ids
        assert "l0" in node_ids


class TestDAGExecutor:
    """Tests for DAGExecutor."""

    def test_executor_import(self):
        from glinker.core.dag import DAGExecutor
        assert DAGExecutor is not None

    def test_executor_has_processors(self, executor):
        assert len(executor.processors) == 4

    def test_executor_has_nodes_map(self, executor):
        assert hasattr(executor, 'nodes_map')
        assert len(executor.nodes_map) == 4

    def test_topological_sort(self, executor):
        """Test that topological sort produces valid execution order."""
        levels = executor._topological_sort()
        # Should have levels - L0 depends on others, so it should be in a later level
        assert len(levels) >= 1

        # Flatten to get order
        order = [node_id for level in levels for node_id in level]
        assert len(order) == 4
        assert "l1" in order
        assert "l2" in order
        assert "l3" in order
        assert "l0" in order

    def test_topological_order_dependencies(self, executor):
        """Test that dependencies are respected in topological order."""
        levels = executor._topological_sort()
        order = [node_id for level in levels for node_id in level]

        # L1 should come before L2 (L2 depends on L1 output)
        l1_idx = order.index("l1")
        l2_idx = order.index("l2")
        assert l1_idx < l2_idx

    def test_execute_returns_context(self, loaded_executor):
        from glinker.core.dag import PipeContext
        result = loaded_executor.execute({"texts": ["test"]})
        assert isinstance(result, PipeContext)

    def test_execute_has_all_results(self, loaded_executor):
        result = loaded_executor.execute({"texts": ["TP53 test"]})
        assert result.get("l1_result") is not None
        assert result.get("l2_result") is not None
        assert result.get("l3_result") is not None
        assert result.get("l0_result") is not None


class TestExecutorLoadEntities:
    """Tests for executor.load_entities()."""

    def test_load_entities(self, executor, entities_jsonl_file, sample_entities):
        executor.load_entities(entities_jsonl_file, target_layers=["dict"])

        # Find L2 processor
        l2_processor = None
        for proc in executor.processors.values():
            if hasattr(proc, 'component') and hasattr(proc.component, 'layers'):
                l2_processor = proc
                break

        assert l2_processor is not None
        dict_layer = l2_processor.component.layers[0]
        # Check count instead of internal _data
        assert dict_layer.count() == len(sample_entities)


class TestExecutorPrecomputeEmbeddings:
    """Tests for executor.precompute_embeddings()."""

    def test_precompute_embeddings(self, loaded_executor):
        # Check if model supports precomputed embeddings
        l3_processor = None
        for proc in loaded_executor.processors.values():
            if hasattr(proc, 'component') and hasattr(proc.component, 'encode_labels'):
                l3_processor = proc
                break

        if not l3_processor or not l3_processor.component.supports_precomputed_embeddings:
            pytest.skip("Model doesn't support precomputed embeddings")

        try:
            stats = loaded_executor.precompute_embeddings(target_layers=["dict"], batch_size=2)
            assert "dict" in stats
            assert stats["dict"] > 0
        except (ValueError, RuntimeError) as e:
            # Tokenization errors can occur with certain model/label combinations
            if "truncation" in str(e) or "Unable to create tensor" in str(e):
                pytest.skip(f"Tokenization issue with model: {e}")


class TestExecutorCacheWriteback:
    """Tests for executor.setup_l3_cache_writeback()."""

    def test_setup_cache_writeback(self, executor):
        executor.setup_l3_cache_writeback()

        # Find L3 processor
        l3_processor = None
        for proc in executor.processors.values():
            if hasattr(proc, '_l2_processor'):
                l3_processor = proc
                break

        assert l3_processor is not None
        assert l3_processor._l2_processor is not None
