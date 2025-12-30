from typing import Any
from src.core.base import BaseProcessor
from src.core.registry import processor_registry
from .models import L1Config, L1Input, L1Output
from .component import L1Component


class L1BatchProcessor(BaseProcessor[L1Config, L1Input, L1Output]):
    """Optimized batch processor using spaCy pipe"""
    
    def __init__(
        self,
        config: L1Config,
        component: L1Component,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
        self._validate_pipeline()
    
    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("extract_entities", {}),
            ("deduplicate", {}),
            ("sort_by_position", {})
        ]
    
    def __call__(self, input_data: L1Input) -> L1Output:
        """Process batch using spaCy's efficient pipe"""
        results = []
        
        for doc, original_text in zip(
            self.component.nlp.pipe(
                input_data.texts, 
                batch_size=self.config.batch_size
            ),
            input_data.texts
        ):
            entities = self._extract_from_doc(doc, original_text)
            
            pipeline_rest = [
                (method, kwargs) 
                for method, kwargs in self.pipeline 
                if method != "extract_entities"
            ]
            
            entities = self._execute_pipeline(entities, pipeline_rest)
            results.append(entities)
        
        return L1Output(entities=results)
    
    def _extract_from_doc(self, doc, text: str) -> list:
        """Extract entities from already processed doc"""
        from .models import L1Entity
        
        entities = []
        for ent in doc.ents:
            left_context, right_context = self.component._get_context(
                text, ent.start_char, ent.end_char
            )
            
            entities.append(L1Entity(
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                left_context=left_context,
                right_context=right_context
            ))
        
        return entities


@processor_registry.register("l1_batch")
def create_l1_batch_processor(config_dict: dict, pipeline: list = None) -> L1BatchProcessor:
    """Factory: creates component + batch processor"""
    config = L1Config(**config_dict)
    component = L1Component(config)
    return L1BatchProcessor(config, component, pipeline)