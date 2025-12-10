from typing import List
from src.core.base import BaseComponent
from src.core.registry import component_registry
from src.l1.models import L1Output
from src.l2.models import L2Input, L2Output, L2Candidate
from src.l3.models import L3Input
from .models import L0Config, L1ToL2Config, L2ToL3Config


class L0BaseComponent(BaseComponent[L0Config]):
    """Base converter component"""
    
    def get_available_methods(self) -> List[str]:
        return ["convert"]


@component_registry.register("l0_l1_to_l2")
class L1ToL2Converter(L0BaseComponent):
    """Converts L1Output to L2Input"""
    
    def __init__(self, config: L1ToL2Config):
        super().__init__(config)
        self.config: L1ToL2Config = config
    
    def convert(self, l1_output: L1Output) -> L2Input:
        """Convert L1Output to L2Input"""
        all_mentions = []
        
        for entities in l1_output.entities:
            mentions = [e.text for e in entities]
            
            if self.config.deduplicate_mentions:
                mentions = self.deduplicate_mentions(mentions)
            
            if self.config.min_mention_length > 0:
                mentions = self.filter_by_length(mentions, self.config.min_mention_length)
            
            all_mentions.extend(mentions)
        
        return L2Input(mentions=all_mentions)
    
    def deduplicate_mentions(self, mentions: List[str]) -> List[str]:
        """Remove duplicate mentions (case-insensitive)"""
        seen = set()
        unique = []
        for mention in mentions:
            mention_lower = mention.lower()
            if mention_lower not in seen:
                unique.append(mention)
                seen.add(mention_lower)
        return unique
    
    def filter_by_length(self, mentions: List[str], min_length: int) -> List[str]:
        """Filter mentions by minimum length"""
        return [m for m in mentions if len(m) >= min_length]


@component_registry.register("l0_l2_to_l3")
class L2ToL3Converter(L0BaseComponent):
    """Converts L2Output to L3Input"""
    
    def __init__(self, config: L2ToL3Config):
        super().__init__(config)
        self.config: L2ToL3Config = config
    
    def convert(
        self, 
        l2_output: L2Output, 
        texts: List[str],
        l1_output: L1Output
    ) -> L3Input:
        """Convert L2Output to L3Input"""
        mentions_per_text = [[e.text for e in entities] for entities in l1_output.entities]
        
        if self.config.flatten_per_text:
            candidates_per_text = self.prepare_per_text(
                l2_output.candidates, 
                mentions_per_text
            )
        else:
            all_candidates = self.flatten_candidates(l2_output.candidates)
            if self.config.deduplicate_candidates:
                all_candidates = self.deduplicate_candidates(all_candidates)
            candidates_per_text = [all_candidates] * len(texts)
        
        return L3Input(texts=texts, candidates=candidates_per_text)
    
    def flatten_candidates(self, candidates: List[List[L2Candidate]]) -> List[L2Candidate]:
        """Flatten nested candidate lists"""
        flat = []
        for candidate_list in candidates:
            flat.extend(candidate_list)
        return flat
    
    def deduplicate_candidates(self, candidates: List[L2Candidate]) -> List[L2Candidate]:
        """Remove duplicate candidates by entity_id"""
        seen = set()
        unique = []
        for c in candidates:
            if c.entity_id not in seen:
                unique.append(c)
                seen.add(c.entity_id)
        return unique
    
    def prepare_per_text(
        self,
        all_candidates: List[List[L2Candidate]],
        mentions_per_text: List[List[str]]
    ) -> List[List[L2Candidate]]:
        """Prepare candidates per text based on mention structure"""
        candidates_per_text = []
        start_idx = 0
        
        for mentions in mentions_per_text:
            num_mentions = len(mentions)
            text_candidates = all_candidates[start_idx:start_idx + num_mentions]
            
            flat = self.flatten_candidates(text_candidates)
            
            if self.config.deduplicate_candidates:
                flat = self.deduplicate_candidates(flat)
            
            candidates_per_text.append(flat)
            start_idx += num_mentions
        
        return candidates_per_text