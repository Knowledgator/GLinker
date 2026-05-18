"""
GLiNKER - Entity Linking Framework
A modular 4-layer entity linking pipeline using spaCy NER, database search, and GLiNER.
"""

__version__ = "0.1.1"

from glinker.l0 import processor as _l0_processor
from glinker.l1 import processor as _l1_processor
from glinker.l2 import processor as _l2_processor
from glinker.l3 import processor as _l3_processor
from glinker.l4 import processor as _l4_processor
from glinker.core import (
    PipeNode,
    BaseInput,
    BaseConfig,
    BaseOutput,
    DAGExecutor,
    DAGPipeline,
    InputConfig,
    PipeContext,
    OutputConfig,
    BaseComponent,
    BaseProcessor,
    ConfigBuilder,
    FieldResolver,
    ReshapeConfig,
    ProcessorFactory,
    ProcessorRegistry,
    load_yaml,
    processor_registry,
)

__all__ = [
    "BaseComponent",
    "BaseConfig",
    "BaseInput",
    "BaseOutput",
    "BaseProcessor",
    "ConfigBuilder",
    "DAGExecutor",
    "DAGPipeline",
    "FieldResolver",
    "InputConfig",
    "OutputConfig",
    "PipeContext",
    "PipeNode",
    "ProcessorFactory",
    "ProcessorRegistry",
    "ReshapeConfig",
    "load_yaml",
    "processor_registry",
]
