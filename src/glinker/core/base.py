from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any
from pydantic import BaseModel, Field


ConfigT = TypeVar('ConfigT', bound=BaseModel)
InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT', bound=BaseModel)


class BaseConfig(BaseModel):
    """Base configuration for all components"""
    pass


class BaseInput(BaseModel):
    """Base input model"""
    pass


class BaseOutput(BaseModel):
    """Base output model"""
    pass


class BaseComponent(ABC, Generic[ConfigT]):
    """
    Base component class that implements core logic.
    Each component should have discrete methods that can be chained.
    """
    
    def __init__(self, config: ConfigT):
        self.config = config
        self._setup()
    
    def _setup(self):
        """Override this for initialization logic"""
        pass
    
    @abstractmethod
    def get_available_methods(self) -> list[str]:
        """Return list of available pipeline methods"""
        pass


class BaseProcessor(ABC, Generic[ConfigT, InputT, OutputT]):
    """
    Base processor that orchestrates component methods via pipeline.
    """
    
    def __init__(
        self,
        config: ConfigT,
        component: BaseComponent[ConfigT],
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        self.config = config
        self.component = component
        self.pipeline = pipeline or self._default_pipeline()
    
    @abstractmethod
    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        """Define default pipeline for this processor"""
        pass
    
    def _validate_pipeline(self):
        """Validate that all pipeline methods exist in component"""
        available = self.component.get_available_methods()
        for method_name, _ in self.pipeline:
            if method_name not in available:
                raise ValueError(
                    f"Method '{method_name}' not found in component. "
                    f"Available: {available}"
                )
    
    def _execute_pipeline_step(
        self, 
        data: Any, 
        method_name: str, 
        kwargs: dict[str, Any]
    ) -> Any:
        """Execute single pipeline step"""
        method = getattr(self.component, method_name)
        return method(data, **kwargs)
    
    def _execute_pipeline(
        self, 
        data: Any, 
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ) -> Any:
        """Execute full pipeline on data"""
        pipe = pipeline or self.pipeline
        result = data
        
        for method_name, kwargs in pipe:
            result = self._execute_pipeline_step(result, method_name, kwargs)
        
        return result
    
    @abstractmethod
    def __call__(self, input_data: InputT) -> OutputT:
        """Process input through pipeline"""
        pass