"""
Tests for src/core/factory.py - ProcessorFactory.
"""

import pytest
import tempfile
import yaml


class TestProcessorFactory:
    """Tests for ProcessorFactory."""

    def test_factory_import(self):
        from src.core.factory import ProcessorFactory
        assert ProcessorFactory is not None

    def test_create_from_dict(self, pipeline_config_dict):
        from src.core.factory import ProcessorFactory
        executor = ProcessorFactory.create_from_dict(pipeline_config_dict)
        assert executor is not None

    def test_create_from_dict_verbose(self, pipeline_config_dict):
        from src.core.factory import ProcessorFactory
        executor = ProcessorFactory.create_from_dict(pipeline_config_dict, verbose=True)
        assert executor is not None

    def test_create_from_dict_returns_executor(self, pipeline_config_dict):
        from src.core.factory import ProcessorFactory
        from src.core.dag import DAGExecutor
        executor = ProcessorFactory.create_from_dict(pipeline_config_dict)
        assert isinstance(executor, DAGExecutor)

    def test_created_executor_has_processors(self, pipeline_config_dict):
        from src.core.factory import ProcessorFactory
        executor = ProcessorFactory.create_from_dict(pipeline_config_dict)
        assert len(executor.processors) == 4

    def test_created_executor_has_nodes_map(self, pipeline_config_dict):
        from src.core.factory import ProcessorFactory
        executor = ProcessorFactory.create_from_dict(pipeline_config_dict)
        assert len(executor.nodes_map) == 4
        assert "l1" in executor.nodes_map
        assert "l2" in executor.nodes_map
        assert "l3" in executor.nodes_map
        assert "l0" in executor.nodes_map

    def test_create_from_file(self, pipeline_config_dict, tmp_path):
        from src.core.factory import ProcessorFactory

        # Write config to temp file
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(pipeline_config_dict, f)

        executor = ProcessorFactory.create_pipeline(str(config_file))
        assert executor is not None

    def test_empty_nodes_config(self):
        from src.core.factory import ProcessorFactory
        config = {"name": "empty", "nodes": []}
        executor = ProcessorFactory.create_from_dict(config)
        assert len(executor.processors) == 0

    def test_invalid_processor_raises(self):
        from src.core.factory import ProcessorFactory
        config = {
            "name": "invalid",
            "nodes": [{
                "id": "test",
                "processor": "nonexistent_processor",
                "inputs": {},
                "output": {"key": "result"},
                "config": {}
            }]
        }
        with pytest.raises(RuntimeError):
            ProcessorFactory.create_from_dict(config)

    def test_missing_nodes_key(self):
        from src.core.factory import ProcessorFactory
        config = {"name": "no_nodes"}
        with pytest.raises(Exception):
            ProcessorFactory.create_from_dict(config)
