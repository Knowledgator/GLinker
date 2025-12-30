from pydantic import Field
from typing import List, Any
from src.core.base import BaseConfig, BaseInput, BaseOutput


class L3Config(BaseConfig):
    model_name: str = Field(...)
    token: str = Field(None)
    device: str = Field("cpu")
    threshold: float = Field(0.5)
    flat_ner: bool = Field(True)
    multi_label: bool = Field(False)
    batch_size: int = Field(8)


# TODO replace candidates with labels
class L3Input(BaseInput):
    texts: List[str] = Field(...)
    candidates: List[List[Any]] = Field(...)


class L3Entity(BaseOutput):
    text: str
    label: str
    start: int
    end: int
    score: float


class L3Output(BaseOutput):
    entities: List[List[L3Entity]] = Field(...)