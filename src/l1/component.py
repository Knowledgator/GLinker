import spacy
from spacy.language import Language
from src.core.base import BaseComponent
from .models import L1Config, L1Entity


class L1Component(BaseComponent[L1Config]):
    """spaCy-based entity extraction component"""
    
    def _setup(self):
        """Initialize spaCy model"""
        self.nlp = self._load_model()
    
    def _load_model(self) -> Language:
        """Load or download spaCy model"""
        try:
            nlp = spacy.load(self.config.model)
            if self.config.device != "cpu":
                spacy.require_gpu()
            return nlp
        except OSError:
            from spacy.cli import download
            download(self.config.model)
            return spacy.load(self.config.model)
    
    def get_available_methods(self) -> list[str]:
        """Return list of available pipeline methods"""
        return [
            "extract_entities",
            "filter_by_length",
            "deduplicate",
            "sort_by_position",
            "add_noun_chunks"
        ]
    
    def extract_entities(self, text: str) -> list[L1Entity]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []
        seen_spans = set()
        
        for ent in doc.ents:
            span = (ent.start_char, ent.end_char)
            if span in seen_spans:
                continue
            
            left_context, right_context = self._get_context(
                text, ent.start_char, ent.end_char
            )
            
            entities.append(L1Entity(
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                left_context=left_context,
                right_context=right_context
            ))
            seen_spans.add(span)
        
        return entities
    
    def filter_by_length(
        self, 
        entities: list[L1Entity], 
        min_length: int = None
    ) -> list[L1Entity]:
        """Filter entities by minimum text length"""
        min_len = min_length if min_length is not None else self.config.min_entity_length
        return [e for e in entities if len(e.text) >= min_len]
    
    def deduplicate(self, entities: list[L1Entity]) -> list[L1Entity]:
        """Remove duplicate entities by span"""
        seen_spans = set()
        unique = []
        
        for entity in entities:
            span = (entity.start, entity.end)
            if span not in seen_spans:
                unique.append(entity)
                seen_spans.add(span)
        
        return unique
    
    def sort_by_position(self, entities: list[L1Entity]) -> list[L1Entity]:
        """Sort entities by start position"""
        return sorted(entities, key=lambda x: x.start)
    
    def add_noun_chunks(
        self, 
        text: str, 
        entities: list[L1Entity] = None
    ) -> list[L1Entity]:
        """Add noun chunks to entities list"""
        if entities is None:
            entities = []
        
        doc = self.nlp(text)
        seen_spans = {(e.start, e.end) for e in entities}
        
        for chunk in doc.noun_chunks:
            span = (chunk.start_char, chunk.end_char)
            
            overlap = False
            for (s, e) in seen_spans:
                if not (chunk.end_char <= s or chunk.start_char >= e):
                    overlap = True
                    break
            
            if not overlap and len(chunk.text) >= self.config.min_entity_length:
                left_context, right_context = self._get_context(
                    text, chunk.start_char, chunk.end_char
                )
                
                entities.append(L1Entity(
                    text=chunk.text,
                    start=chunk.start_char,
                    end=chunk.end_char,
                    left_context=left_context,
                    right_context=right_context
                ))
                seen_spans.add(span)
        
        return entities
    
    def _get_context(self, text: str, start: int, end: int) -> tuple[str, str]:
        """Extract left and right context for entity"""
        left_start = max(0, start - self.config.max_left_context)
        left_context = text[left_start:start].strip()
        
        right_end = min(len(text), end + self.config.max_right_context)
        right_context = text[end:right_end].strip()
        
        return left_context, right_context