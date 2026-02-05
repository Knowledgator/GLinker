from pydantic import Field
from typing import List, Optional
from glinker.core.base import BaseConfig, BaseInput, BaseOutput


class L1Config(BaseConfig):
    model: str = Field("en_core_sci_sm", description="spaCy model identifier")
    device: str = Field("cpu", description="Device to run the model on")
    batch_size: int = Field(16, description="Batch size for processing")
    max_right_context: int = Field(50, description="Maximum right context length")
    max_left_context: int = Field(50, description="Maximum left context length")
    min_entity_length: int = Field(2, description="Minimum entity text length")
    include_noun_chunks: bool = Field(False, description="Include noun chunks")


class L1GlinerConfig(L1Config):
    """Configuration for GLiNER-based L1 entity extraction"""
    model: str = Field(..., description="GLiNER model identifier (overrides spaCy model)")
    labels: List[str] = Field(..., description="Fixed list of labels for entity extraction")
    token: Optional[str] = Field(None, description="HuggingFace token")
    threshold: float = Field(0.3, description="Confidence threshold for entity extraction")
    flat_ner: bool = Field(True, description="Use flat NER (no nested entities)")
    multi_label: bool = Field(False, description="Allow multiple labels per entity")
    use_precomputed_embeddings: bool = Field(
        False,
        description="Use precomputed label embeddings (BiEncoder only)"
    )
    max_length: Optional[int] = Field(
        None,
        description="Maximum sequence length for tokenization"
    )


class L1Input(BaseInput):
    texts: list[str] = Field(..., description="List of text inputs")


class L1Entity(BaseOutput):
    text: str = Field(..., description="Extracted mention text")
    label: Optional[str] = Field(None, description="Entity label/type")
    start: int = Field(..., description="Start position")
    end: int = Field(..., description="End position")
    left_context: str = Field(..., description="Left context")
    right_context: str = Field(..., description="Right context")


class L1Output(BaseOutput):
    entities: list[list[L1Entity]] = Field(..., description="Extracted entities per text")