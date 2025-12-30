from typing import Type, Dict, Any, Callable
from .base import BaseProcessor, BaseComponent


class ComponentRegistry:

    def __init__(self):
        self._registry: Dict[str, Type[BaseComponent]] = {}
    
    def register(self, name: str):
        def decorator(cls: Type[BaseComponent]):
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type[BaseComponent]:
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def list_available(self) -> list[str]:
        return list(self._registry.keys())


class ProcessorRegistry:    
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
    
    def register(self, name: str):
        def decorator(factory: Callable):
            self._registry[name] = factory
            return factory
        return decorator
    
    def get(self, name: str) -> Callable:
        if name not in self._registry:
            raise KeyError(f"Processor '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def list_available(self) -> list[str]:
        return list(self._registry.keys())


component_registry = ComponentRegistry()
processor_registry = ProcessorRegistry()