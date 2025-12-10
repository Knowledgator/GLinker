from typing import Any
from src.core.base import BaseProcessor
from src.core.registry import processor_registry, component_registry
from .models import L3Config, L3Input, L3Output, L3Entity
from .component import L3BaseComponent


class L3Processor(BaseProcessor[L3Config, L3Input, L3Output]):
    """Processor for L3 entity extraction with GLiNER (iterative)"""
    
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
        """Process texts with candidates through pipeline (one by one)"""
        all_entities = []
        
        for text, candidates in zip(input_data.texts, input_data.candidates):
            entities = self.component.predict_entities(text, candidates)
            
            for method_name, kwargs in self.pipeline[1:]:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)
            
            all_entities.append(entities)
        
        return L3Output(entities=all_entities)


class L3BatchProcessor(BaseProcessor[L3Config, L3Input, L3Output]):
    """Batch processor - processes texts efficiently based on label similarity"""
    
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
            ("filter_by_score", {}),
            ("sort_by_position", {})
        ]
    
    def __call__(self, input_data: L3Input) -> L3Output:
        """Process texts in batch when possible"""
        
        all_labels = []
        for candidates in input_data.candidates:
            labels = []
            seen = set()
            for c in candidates:
                label_lower = c.label.lower()
                if label_lower not in seen:
                    labels.append(c.label)
                    seen.add(label_lower)
            all_labels.append(labels)
        
        labels_sets = [tuple(sorted(labels)) for labels in all_labels]
        all_same_labels = len(set(labels_sets)) == 1
        
        if all_same_labels and all_labels[0]:
            labels = all_labels[0]
            
            batch_predictions = self.component.model.predict_entities(
                input_data.texts,
                labels,
                threshold=self.config.threshold,
                flat_ner=self.config.flat_ner,
                multi_label=self.config.multi_label
            )
        else:
            batch_predictions = []
            for text, labels in zip(input_data.texts, all_labels):
                if not labels:
                    batch_predictions.append([])
                    continue
                
                predictions = self.component.model.predict_entities(
                    text,
                    labels,
                    threshold=self.config.threshold,
                    flat_ner=self.config.flat_ner,
                    multi_label=self.config.multi_label
                )
                batch_predictions.append(predictions)
        
        all_entities = []
        for predictions in batch_predictions:
            entities = []
            for pred in predictions:
                entities.append(L3Entity(
                    text=pred['text'],
                    start=pred['start'],
                    end=pred['end'],
                    label=pred['label'],
                    score=pred['score']
                ))
            
            for method_name, kwargs in self.pipeline:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)
            
            all_entities.append(entities)
        
        return L3Output(entities=all_entities)


@processor_registry.register("l3_gliner")
def create_l3_gliner_processor(config_dict: dict, pipeline: list = None) -> L3Processor:
    """Factory: L3 processor with GLiNER component (iterative)"""
    config = L3Config(**config_dict)
    component_cls = component_registry.get("l3_gliner")
    component = component_cls(config)
    return L3Processor(config, component, pipeline)


@processor_registry.register("l3_batch")
def create_l3_batch_processor(config_dict: dict, pipeline: list = None) -> L3BatchProcessor:
    """Factory: Batch L3 processor - batches when labels are same"""
    config = L3Config(**config_dict)
    component_cls = component_registry.get("l3_gliner")
    component = component_cls(config)
    return L3BatchProcessor(config, component, pipeline)


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