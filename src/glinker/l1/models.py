from pydantic import Field
from glinker.core.base import BaseConfig, BaseInput, BaseOutput


class L1Config(BaseConfig):
    model: str = Field("en_core_sci_sm", description="spaCy model identifier")
    device: str = Field("cpu", description="Device to run the model on")
    batch_size: int = Field(16, description="Batch size for processing")
    max_right_context: int = Field(50, description="Maximum right context length")
    max_left_context: int = Field(50, description="Maximum left context length")
    min_entity_length: int = Field(2, description="Minimum entity text length")
    include_noun_chunks: bool = Field(False, description="Include noun chunks")


class L1Input(BaseInput):
    texts: list[str] = Field(..., description="List of text inputs")


class L1Entity(BaseOutput):
    text: str = Field(..., description="Extracted mention text")
    start: int = Field(..., description="Start position")
    end: int = Field(..., description="End position")
    left_context: str = Field(..., description="Left context")
    right_context: str = Field(..., description="Right context")


class L1Output(BaseOutput):
    entities: list[list[L1Entity]] = Field(..., description="Extracted entities per text")