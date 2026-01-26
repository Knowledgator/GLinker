"""
GLiNKER - Entity Linking Framework
A modular 4-layer entity linking pipeline using spaCy NER, database search, and GLiNER.
"""

__version__ = "0.1.0"

# Import processors to register them (decorators will execute)
from glinker.l0 import processor as _l0_processor
from glinker.l1 import processor as _l1_processor
from glinker.l2 import processor as _l2_processor
from glinker.l3 import processor as _l3_processor

# Import core
from glinker.core import (
    BaseConfig,
    BaseInput,
    BaseOutput,
    BaseComponent,
    BaseProcessor,
    ProcessorRegistry,
    processor_registry,
    ProcessorFactory,
    load_yaml,
    InputConfig,
    OutputConfig,
    ReshapeConfig,
    PipeNode,
    PipeContext,
    FieldResolver,
    DAGPipeline,
    DAGExecutor,
)

__all__ = [
    'BaseConfig',
    'BaseInput',
    'BaseOutput',
    'BaseComponent',
    'BaseProcessor',
    'ProcessorRegistry',
    'processor_registry',
    'ProcessorFactory',
    'load_yaml',
    'InputConfig',
    'OutputConfig',
    'ReshapeConfig',
    'PipeNode',
    'PipeContext',
    'FieldResolver',
    'DAGPipeline',
    'DAGExecutor',
]
