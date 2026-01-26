"""
L0 - Aggregation Layer

Combines outputs from L1 (mention extraction), L2 (candidate retrieval),
and L3 (entity linking) into unified L0Entity structures with full pipeline context.
"""

from .models import L0Config, L0Input, L0Output, L0Entity, LinkedEntity
from .component import L0Component
from .processor import L0Processor, create_l0_processor

__all__ = [
    "L0Config",
    "L0Input",
    "L0Output",
    "L0Entity",
    "LinkedEntity",
    "L0Component",
    "L0Processor",
    "create_l0_processor"
]
