"""
Tests for src/core/base.py - Base classes for components and processors.
"""

import pytest
from pydantic import ValidationError


class TestBaseConfig:
    """Tests for BaseConfig."""

    def test_base_config_import(self):
        from src.core.base import BaseConfig
        assert BaseConfig is not None

    def test_base_config_inheritance(self):
        from src.core.base import BaseConfig

        class CustomConfig(BaseConfig):
            custom_field: str = "default"

        config = CustomConfig()
        assert config.custom_field == "default"

    def test_base_config_validation(self):
        from src.core.base import BaseConfig

        class CustomConfig(BaseConfig):
            required_field: str

        with pytest.raises(ValidationError):
            CustomConfig()  # Missing required field


class TestBaseInput:
    """Tests for BaseInput."""

    def test_base_input_import(self):
        from src.core.base import BaseInput
        assert BaseInput is not None


class TestBaseOutput:
    """Tests for BaseOutput."""

    def test_base_output_import(self):
        from src.core.base import BaseOutput
        assert BaseOutput is not None


class TestBaseComponent:
    """Tests for BaseComponent."""

    def test_base_component_import(self):
        from src.core.base import BaseComponent
        assert BaseComponent is not None

    def test_base_component_abstract(self):
        from src.core.base import BaseComponent, BaseConfig

        class TestConfig(BaseConfig):
            pass

        # BaseComponent requires get_available_methods implementation
        class TestComponent(BaseComponent[TestConfig]):
            def get_available_methods(self):
                return ["test_method"]

        config = TestConfig()
        component = TestComponent(config)
        assert component.get_available_methods() == ["test_method"]


class TestBaseProcessor:
    """Tests for BaseProcessor."""

    def test_base_processor_import(self):
        from src.core.base import BaseProcessor
        assert BaseProcessor is not None

    def test_base_processor_pipeline_validation(self):
        from src.core.base import BaseProcessor, BaseComponent, BaseConfig

        class TestConfig(BaseConfig):
            pass

        class TestComponent(BaseComponent[TestConfig]):
            def get_available_methods(self):
                return ["method1", "method2"]

            def method1(self, data):
                return data

            def method2(self, data):
                return data

        class TestProcessor(BaseProcessor):
            def _default_pipeline(self):
                return [("method1", {}), ("method2", {})]

            def __call__(self, input_data):
                return self._execute_pipeline(input_data)

        config = TestConfig()
        component = TestComponent(config)
        processor = TestProcessor(config, component)

        assert processor.pipeline is not None
        # Test pipeline execution
        result = processor("test_input")
        assert result == "test_input"
