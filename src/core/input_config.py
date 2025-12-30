from pydantic import BaseModel, Field
from typing import Literal, Union, List, Any, Optional


class ReshapeConfig(BaseModel):
    """Configuration for data reshaping"""
    by: str = Field(..., description="Reference structure path: 'l1_result.entities'")
    mode: Literal["flatten_per_group", "preserve_structure"] = Field(
        "flatten_per_group",
        description="Reshape mode"
    )


class InputConfig(BaseModel):
    """
    Unified input data specification
    
    Examples:
```yaml
    source: "l1_result"
    fields: "entities[*].text"
    reduce: "flatten"
    
    source: "$input"
    fields: "texts"
    
    source: "outputs[-1]"
    fields: ["label", "score"]
```
    """
    source: str = Field(
        ...,
        description="Data source: key ('l1_result'), index ('outputs[-1]'), or '$input'"
    )
    
    fields: Union[str, List[str], None] = Field(
        None,
        description="JSONPath fields: 'entities[*].text' or ['label', 'score']"
    )
    
    reduce: Literal["all", "first", "last", "flatten"] = Field(
        "all",
        description="Reduction mode for lists"
    )
    
    reshape: Optional[ReshapeConfig] = Field(
        None,
        description="Data reshaping configuration"
    )
    
    template: Optional[str] = Field(
        None,
        description="Field concatenation template: '{label}: {description}'"
    )
    
    filter: Optional[str] = Field(
        None,
        description="Filter expression: 'score > 0.5'"
    )
    
    default: Any = None


class OutputConfig(BaseModel):
    """Output specification"""
    key: str = Field(..., description="Key for storing in context")
    fields: Union[str, List[str], None] = Field(
        None,
        description="Fields to save (optional, defaults to all)"
    )