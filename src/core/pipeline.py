from typing import List, Any, Dict, Optional
import yaml
import logging
from .registry import processor_registry


logger = logging.getLogger(__name__)


class PipelineDict:
    """Context for passing data between pipeline steps"""
    
    def __init__(self):
        self.data = {}
    
    def set(self, key: str, value: Any):
        self.data[key] = value
    
    def get(self, key: str) -> Any:
        return self.data.get(key)


class PipelineStep:
    """Single step in pipeline"""
    
    def __init__(
        self, 
        processor_name: str, 
        config: dict,
        inputs: Dict[str, str],
        output: str
    ):
        self.processor_name = processor_name
        self.config = config
        self.inputs = inputs
        self.output = output
        self.processor = None
    
    def setup(self):
        """Initialize processor from registry"""
        self.processor = processor_registry.get(self.processor_name)(
            config_dict=self.config
        )
    
    def execute(self, context: PipelineDict) -> Any:
        """Execute step with context"""
        kwargs = {}
        for param_name, context_key in self.inputs.items():
            value = context.get(context_key)
            if value is None:
                raise ValueError(
                    f"Processor '{self.processor_name}': Input '{context_key}' not found in context"
                )
            kwargs[param_name] = value
        
        if len(kwargs) == 1:
            result = self.processor(list(kwargs.values())[0])
        else:
            result = self.processor(**kwargs)
        
        context.set(self.output, result)
        
        return result


class Pipeline:
    """Universal pipeline orchestrator"""
    
    def __init__(
        self, 
        name: str, 
        steps: List[PipelineStep],
        verbose: bool = False
    ):
        self.name = name
        self.steps = steps
        self.verbose = verbose
        self._setup_processors()
    
    def _setup_processors(self):
        """Setup all processors"""
        if self.verbose:
            logger.info(f"Setting up pipeline: {self.name}")
        for i, step in enumerate(self.steps, 1):
            if self.verbose:
                logger.info(f"  Step {i}: {step.processor_name}")
            step.setup()
        if self.verbose:
            logger.info("Pipeline ready")
    
    @classmethod
    def from_yaml(cls, config_path: str, verbose: bool = False) -> 'Pipeline':
        """Load pipeline from YAML config"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        pipeline_config = config['pipeline']
        name = pipeline_config['name']
        
        steps = []
        for step_config in pipeline_config['steps']:
            step = PipelineStep(
                processor_name=step_config['processor'],
                config=step_config.get('config', {}),
                inputs=step_config['inputs'],
                output=step_config['output']
            )
            steps.append(step)
        
        return cls(name, steps, verbose=verbose)
    
    def __call__(self, l1_input) -> PipelineDict:
        """Run full pipeline and return PipelineDict with all outputs"""
        context = PipelineDict()
        context.set('l1_input', l1_input)
        context.set('texts', l1_input.texts)
        
        for i, step in enumerate(self.steps, 1):
            if self.verbose:
                logger.info(f"Step {i}/{len(self.steps)}: {step.processor_name}")
            
            step.execute(context)
            
            if self.verbose:
                logger.info(f"  ✓ Output stored as '{step.output}'")
        
        return context