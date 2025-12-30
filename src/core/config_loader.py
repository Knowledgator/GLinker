import yaml
from pathlib import Path
from typing import Any


class ConfigLoader:
    """Load configurations from YAML files"""
    
    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        """Load YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
        