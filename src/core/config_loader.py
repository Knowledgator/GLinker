import yaml
from pathlib import Path
from typing import Type, Any
from pydantic import BaseModel


class ConfigLoader:
    """Load configurations from YAML files"""
    
    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        """Load YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_config(
        config_data: dict[str, Any],
        config_class: Type[BaseModel]
    ) -> BaseModel:
        """Parse config dict into Pydantic model"""
        return config_class(**config_data)
    
    @staticmethod
    def load_pipeline(pipeline_data: list[dict]) -> list[tuple[str, dict]]:
        """Parse pipeline from YAML format"""
        return [
            (step['method'], step.get('kwargs', {}))
            for step in pipeline_data
        ]
    
    @classmethod
    def load_processor_config(
        cls,
        path: str | Path,
        processor_key: str = None
    ) -> dict[str, Any]:
        """
        Load full processor configuration from YAML.
        
        Returns dict with:
        - config: dict for component config
        - pipeline: list of (method, kwargs) tuples
        """
        data = cls.load_yaml(path)
        
        if processor_key:
            data = data[processor_key]
        
        return {
            'config': data.get('config', {}),
            'pipeline': cls.load_pipeline(data.get('pipeline', []))
        }


class PresetLoader:
    """Load preset configurations"""
    
    def __init__(self, presets_path: str | Path):
        self.presets_path = Path(presets_path)
        self.presets = ConfigLoader.load_yaml(presets_path).get('presets', {})
    
    def load_preset(self, preset_name: str) -> dict[str, Any]:
        """Load specific preset"""
        if preset_name not in self.presets:
            raise KeyError(
                f"Preset '{preset_name}' not found. "
                f"Available: {list(self.presets.keys())}"
            )
        
        preset_data = self.presets[preset_name]
        return {
            'config': preset_data.get('config', {}),
            'pipeline': ConfigLoader.load_pipeline(preset_data.get('pipeline', []))
        }
    
    def list_presets(self) -> list[str]:
        """List available presets"""
        return list(self.presets.keys())