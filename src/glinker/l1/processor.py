from typing import Any, List, Union
from glinker.core.base import BaseProcessor
from glinker.core.registry import processor_registry
from .models import L1Config, L1GlinerConfig, L1Input, L1Output
from .component import L1SpacyComponent, L1GlinerComponent


class L1SpacyProcessor(BaseProcessor[L1Config, L1Input, L1Output]):
    """Optimized batch processor using spaCy pipe"""

    def __init__(
        self,
        config: L1Config,
        component: L1SpacyComponent,
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
    
    def __call__(
        self, 
        texts: List[str] = None,
        input_data: L1Input = None
    ) -> L1Output:
        """Process batch using spaCy's efficient pipe"""
        
        # Support both direct texts and L1Input
        if texts is not None:
            texts_to_process = texts
        elif input_data is not None:
            texts_to_process = input_data.texts
        else:
            raise ValueError("Either 'texts' or 'input_data' must be provided")
        
        results = []
        
        for doc, original_text in zip(
            self.component.nlp.pipe(
                texts_to_process, 
                batch_size=self.config.batch_size
            ),
            texts_to_process
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


@processor_registry.register("l1_spacy")
def create_l1_spacy_processor(config_dict: dict, pipeline: list = None) -> L1SpacyProcessor:
    """Factory: creates component + batch processor"""
    config = L1Config(**config_dict)
    component = L1SpacyComponent(config)
    return L1SpacyProcessor(config, component, pipeline)


class L1GlinerProcessor(BaseProcessor[L1GlinerConfig, L1Input, L1Output]):
    """GLiNER-based batch processor for L1 entity extraction"""

    def __init__(
        self,
        config: L1GlinerConfig,
        component: L1GlinerComponent,
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

    def __call__(
        self,
        texts: List[str] = None,
        input_data: L1Input = None
    ) -> L1Output:
        """Process batch of texts using GLiNER"""

        # Support both direct texts and L1Input
        if texts is not None:
            texts_to_process = texts
        elif input_data is not None:
            texts_to_process = input_data.texts
        else:
            raise ValueError("Either 'texts' or 'input_data' must be provided")

        results = []

        # Process each text individually
        for text in texts_to_process:
            # Extract entities using component
            entities = self.component.extract_entities(text)

            # Apply rest of pipeline (skip extract_entities as already done)
            pipeline_rest = [
                (method, kwargs)
                for method, kwargs in self.pipeline
                if method != "extract_entities"
            ]

            entities = self._execute_pipeline(entities, pipeline_rest)
            results.append(entities)

        return L1Output(entities=results)


@processor_registry.register("l1_gliner")
def create_l1_gliner_processor(config_dict: dict, pipeline: list = None) -> L1GlinerProcessor:
    """Factory: creates component + GLiNER processor"""
    config = L1GlinerConfig(**config_dict)
    component = L1GlinerComponent(config)
    return L1GlinerProcessor(config, component, pipeline)