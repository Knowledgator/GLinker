from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from .input_config import InputConfig, OutputConfig


class PipeNode(BaseModel):
    """
    Single node in DAG pipeline
    
    Represents one processing stage with:
    - Inputs (where to get data)
    - Processor (what to do)
    - Output (where to store result)
    - Dependencies (execution order)
    
    Example:
    ```yaml
        id: "l1_extraction"
        processor: "my_l1"
        inputs:
        texts:
            source: "$input"
            fields: "texts"
        output:
        key: "l1_result"
    ```
    """
    
    id: str = Field(..., description="Unique node identifier")
    
    processor: str = Field(..., description="Processor name from registry")
    
    inputs: Dict[str, InputConfig] = Field(
        default_factory=dict,
        description="Input parameter mappings"
    )
    
    output: OutputConfig = Field(..., description="Output specification")
    
    requires: List[str] = Field(
        default_factory=list,
        description="Explicit dependencies (node IDs)"
    )
    
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processor configuration"
    )
    
    schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema for field mappings/transformations"
    )
    
    condition: Optional[str] = Field(
        None,
        description="Conditional execution expression"
    )
    
    class Config:
        fields = {'schema': 'schema'}