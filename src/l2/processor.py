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
        mentions: Union[List[str], List[List[Any]], L2Input] = None,
        structure: List[List[str]] = None,
        input_data: L2Input = None
    ) -> L2Output:
        """
        Process mentions and return candidates
        
        Supports:
        - List[str]: flat list of mention strings
        - List[List[L1Entity]]: nested list of L1Entity objects (one list per text)
        - L2Input: structured input with mentions and structure
        """
        
        if input_data is not None:
            mentions = input_data.mentions
            structure = input_data.structure
        elif isinstance(mentions, L2Input):
            structure = mentions.structure
            mentions = mentions.mentions
        
        # Check if mentions is nested (list of lists - one per text)
        if mentions and isinstance(mentions[0], (list, tuple)):
            # Nested structure: [[entities_text1], [entities_text2], ...]
            all_candidates = []
            
            for text_entities in mentions:
                text_candidates = []
                
                for entity in text_entities:
                    # Extract text from L1Entity or dict
                    mention_text = self._extract_mention_text(entity)
                    
                    # Search candidates for this mention
                    candidates = self._execute_pipeline(mention_text, self.pipeline)
                    text_candidates.extend(candidates)
                
                all_candidates.append(text_candidates)
            
            return L2Output(candidates=all_candidates)
        
        # Flat structure: ["mention1", "mention2", ...]
        else:
            all_candidates = []
            
            for mention in mentions:
                mention_text = self._extract_mention_text(mention)
                candidates = self._execute_pipeline(mention_text, self.pipeline)
                all_candidates.append(candidates)
            
            if structure:
                grouped = self._group_by_structure(all_candidates, structure)
            else:
                # Flatten all into one group
                grouped = [self._flatten(all_candidates)]
            
            return L2Output(candidates=grouped)
    
    def _extract_mention_text(self, mention: Any) -> str:
        """Extract text string from mention (can be L1Entity, dict, or str)"""
        if isinstance(mention, str):
            return mention
        elif hasattr(mention, 'text'):
            return mention.text
        elif isinstance(mention, dict):
            return mention.get('text', str(mention))
        else:
            return str(mention)
    
    def _group_by_structure(
        self,
        all_candidates: List[List[DatabaseRecord]],
        structure: List[List[str]]
    ) -> List[List[DatabaseRecord]]:
        """Group candidates according to structure"""
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
        """Flatten nested list"""
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