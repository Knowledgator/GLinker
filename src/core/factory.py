from typing import Type
from pathlib import Path
from .base import BaseProcessor, BaseComponent, BaseConfig
from .config_loader import ConfigLoader, PresetLoader
from .registry import component_registry, processor_registry


class ProcessorFactory:
    """Factory for creating processors from configs"""
    
    @staticmethod
    def create_from_config(
        component_class: Type[BaseComponent],
        processor_class: Type[BaseProcessor],
        config_class: Type[BaseConfig],
        config_path: str | Path,
        processor_key: str = None
    ) -> BaseProcessor:
        """Create processor from YAML config file"""
        
        loaded = ConfigLoader.load_processor_config(config_path, processor_key)
        
        config = config_class(**loaded['config'])
        component = component_class(config)
        processor = processor_class(
            config=config,
            component=component,
            pipeline=loaded['pipeline'] if loaded['pipeline'] else None
        )
        
        return processor
    
    @staticmethod
    def create_from_preset(
        component_class: Type[BaseComponent],
        processor_class: Type[BaseProcessor],
        config_class: Type[BaseConfig],
        presets_path: str | Path,
        preset_name: str
    ) -> BaseProcessor:
        """Create processor from preset"""
        
        loader = PresetLoader(presets_path)
        preset_data = loader.load_preset(preset_name)
        
        config = config_class(**preset_data['config'])
        component = component_class(config)
        processor = processor_class(
            config=config,
            component=component,
            pipeline=preset_data['pipeline'] if preset_data['pipeline'] else None
        )
        
        return processor
    
    @staticmethod
    def create_from_registry(
        processor_name: str,
        config_dict: dict,
        pipeline: list[tuple[str, dict]] = None
    ) -> BaseProcessor:
        """Create processor using processor registry"""
        factory = processor_registry.get(processor_name)
        return factory(config_dict, pipeline)