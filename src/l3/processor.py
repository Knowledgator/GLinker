from typing import Any
from src.core.base import BaseProcessor
from src.core.registry import processor_registry, component_registry
from .models import L3Config, L3Input, L3Output
from .component import L3BaseComponent


class L3Processor(BaseProcessor[L3Config, L3Input, L3Output]):
    """Processor for L3 entity extraction with GLiNER"""
    
    def __init__(
        self,
        config: L3Config,
        component: L3BaseComponent,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
        self._validate_pipeline()
    
    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("predict_entities", {}),
            ("filter_by_score", {}),
            ("sort_by_position", {})
        ]
    
    def __call__(self, input_data: L3Input) -> L3Output:
        """Process texts with candidates through pipeline"""
        all_entities = []
        
        for text, candidates in zip(input_data.texts, input_data.candidates):
            entities = self.component.predict_entities(text, candidates)
            
            for method_name, kwargs in self.pipeline[1:]:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)
            
            all_entities.append(entities)
        
        return L3Output(entities=all_entities)


# Register processor factories
@processor_registry.register("l3_gliner")
def create_l3_gliner_processor(config_dict: dict, pipeline: list = None) -> L3Processor:
    """Factory: L3 processor with GLiNER component"""
    config = L3Config(**config_dict)
    component_cls = component_registry.get("l3_gliner")
    component = component_cls(config)
    return L3Processor(config, component, pipeline)


@processor_registry.register("l3_strict")
def create_l3_strict_processor(config_dict: dict, pipeline: list = None) -> L3Processor:
    """Factory: Strict mode with high threshold"""
    config = L3Config(**config_dict)
    config.threshold = 0.7
    component_cls = component_registry.get("l3_gliner")
    component = component_cls(config)
    
    default_pipeline = [
        ("predict_entities", {}),
        ("filter_by_score", {"min_score": 0.7}),
        ("sort_by_score", {})
    ]
    
    return L3Processor(config, component, pipeline or default_pipeline)