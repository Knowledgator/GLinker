from typing import Any
from src.core.base import BaseProcessor
from src.core.registry import processor_registry, component_registry
from .models import L2Config, L2Input, L2Output
from .component import L2BaseComponent


class L2Processor(BaseProcessor[L2Config, L2Input, L2Output]):
    
    def __init__(
        self,
        config: L2Config,
        component: L2BaseComponent,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
        self._validate_pipeline()
    
    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("search_exact", {}),
            ("filter_by_popularity", {}),
            ("deduplicate_candidates", {}),
            ("limit_candidates", {})
        ]
    
    def __call__(self, input_data: L2Input) -> L2Output:
        all_candidates = []
        
        for mention in input_data.mentions:
            candidates = self._execute_pipeline(mention, self.pipeline)
            all_candidates.append(candidates)
        
        return L2Output(candidates=all_candidates)


@processor_registry.register("l2_postgres")
def create_l2_postgres_processor(config_dict: dict, pipeline: list = None) -> L2Processor:
    config = L2Config(**config_dict)
    component_cls = component_registry.get("l2_postgres")
    component = component_cls(config)
    return L2Processor(config, component, pipeline)


@processor_registry.register("l2_elasticsearch")
def create_l2_elasticsearch_processor(config_dict: dict, pipeline: list = None) -> L2Processor:
    config = L2Config(**config_dict)
    component_cls = component_registry.get("l2_elasticsearch")
    component = component_cls(config)
    return L2Processor(config, component, pipeline)

@processor_registry.register("l2_redis")
def create_l2_redis_processor(config_dict: dict, pipeline: list = None) -> L2Processor:
    """Factory: L2 processor with Redis component"""
    config = L2Config(**config_dict)
    component_cls = component_registry.get("l2_redis")
    component = component_cls(config)
    return L2Processor(config, component, pipeline)