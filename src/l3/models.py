from pydantic import Field
from typing import List
from src.core.base import BaseConfig, BaseInput, BaseOutput
from src.l2.models import L2Candidate


class L3Config(BaseConfig):
    model_name: str = Field("urchade/gliner_medium-v2.1", description="GLiNER model name")
    device: str = Field("cpu", description="Device to run model on")
    threshold: float = Field(0.5, description="Confidence threshold")
    max_width: int = Field(12, description="Maximum entity width in tokens")
    flat_ner: bool = Field(True, description="Flat NER (no overlapping entities)")
    multi_label: bool = Field(False, description="Multi-label classification")


class L3Input(BaseInput):
    texts: List[str] = Field(..., description="Input texts")
    candidates: List[List[L2Candidate]] = Field(..., description="Candidates from L2 per text")


class L3Entity(BaseOutput):
    text: str = Field(..., description="Entity text")
    start: int = Field(..., description="Start position")
    end: int = Field(..., description="End position")
    label: str = Field(..., description="Predicted label (candidate name)")
    score: float = Field(..., description="Confidence score")


class L3Output(BaseOutput):
    entities: List[List[L3Entity]] = Field(..., description="Predicted entities per text")