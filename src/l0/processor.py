from typing import Any
from src.core.base import BaseProcessor
from src.core.registry import processor_registry, component_registry
from .models import L0Config, L1ToL2Config, L2ToL3Config
from .component import L0BaseComponent


class L0Processor(BaseProcessor[L0Config, Any, Any]):
    """Generic converter processor"""
    
    def __init__(
        self,
        config: L0Config,
        component: L0BaseComponent,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
    
    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [("convert", {})]
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute conversion - flexible args"""
        if args:
            return self.component.convert(args[0], **kwargs)
        else:
            return self.component.convert(**kwargs)


@processor_registry.register("l0_l1_to_l2")
def create_l1_to_l2_converter(config_dict: dict, pipeline: list = None):
    """Factory: L1 to L2 converter"""
    config = L1ToL2Config(**config_dict)
    component_cls = component_registry.get("l0_l1_to_l2")
    component = component_cls(config)
    return L0Processor(config, component, pipeline)


@processor_registry.register("l0_l2_to_l3")
def create_l2_to_l3_converter(config_dict: dict, pipeline: list = None):
    """Factory: L2 to L3 converter"""
    config = L2ToL3Config(**config_dict)
    component_cls = component_registry.get("l0_l2_to_l3")
    component = component_cls(config)
    return L0Processor(config, component, pipeline)