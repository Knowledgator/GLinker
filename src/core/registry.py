from typing import Dict, Callable


class ProcessorRegistry:
    """Registry for processor factory functions"""
    
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
    
    def register(self, name: str):
        """Decorator to register processor factory"""
        def decorator(factory: Callable):
            self._registry[name] = factory
            return factory
        return decorator
    
    def get(self, name: str) -> Callable:
        """Get processor factory by name"""
        if name not in self._registry:
            raise KeyError(
                f"Processor '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]
    
    def list_available(self) -> list[str]:
        """List all registered processor names"""
        return list(self._registry.keys())


processor_registry = ProcessorRegistry()