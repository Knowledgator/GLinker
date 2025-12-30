from typing import Any, List, Union
from src.core.base import BaseProcessor
from src.core.registry import processor_registry
from .models import L2Config, L2Input, L2Output, DatabaseRecord
from .component import DatabaseChainComponent


class L2Processor(BaseProcessor[L2Config, L2Input, L2Output]):
    """Multi-layer database search processor"""
    
    def __init__(
        self,
        config: L2Config,
        component: DatabaseChainComponent,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
    
    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("search", {}),
            ("filter_by_popularity", {}),
            ("deduplicate_candidates", {}),
            ("limit_candidates", {}),
            ("sort_by_popularity", {})
        ]
    
    def __call__(
        self, 
        mentions: Union[List[str], L2Input] = None,
        structure: List[List[str]] = None,
        input_data: L2Input = None
    ) -> L2Output:
        if input_data is not None:
            mentions = input_data.mentions
            structure = input_data.structure
        elif isinstance(mentions, L2Input):
            structure = mentions.structure
            mentions = mentions.mentions
        
        all_candidates = []
        for mention in mentions:
            candidates = self._execute_pipeline(mention, self.pipeline)
            all_candidates.append(candidates)
        
        if structure:
            grouped = self._group_by_structure(all_candidates, structure)
        else:
            grouped = [self._flatten(all_candidates)]
        
        return L2Output(candidates=grouped)
    
    def _group_by_structure(
        self,
        all_candidates: List[List[DatabaseRecord]],
        structure: List[List[str]]
    ) -> List[List[DatabaseRecord]]:
        grouped = []
        idx = 0
        for text_mentions in structure:
            text_candidates = []
            for _ in text_mentions:
                if idx < len(all_candidates):
                    text_candidates.extend(all_candidates[idx])
                    idx += 1
            grouped.append(text_candidates)
        return grouped
    
    def _flatten(self, nested: List[List[Any]]) -> List[Any]:
        flat = []
        for sublist in nested:
            flat.extend(sublist)
        return flat


@processor_registry.register("l2_chain")
def create_l2_processor(config_dict: dict, pipeline: list = None) -> L2Processor:
    """Factory: creates component + processor"""
    config = L2Config(**config_dict)
    component = DatabaseChainComponent(config)
    return L2Processor(config, component, pipeline)