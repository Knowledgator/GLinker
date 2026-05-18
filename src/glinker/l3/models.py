from typing import Any

from pydantic import Field

from glinker.core.base import BaseInput, BaseConfig, BaseOutput


class L3Config(BaseConfig):
    model_name: str = Field(...)
    token: str | None = Field(None)
    device: str = Field("cpu")
    threshold: float = Field(0.5)
    flat_ner: bool = Field(True)
    multi_label: bool = Field(False)
    batch_size: int = Field(8)

    # Embedding settings
    use_precomputed_embeddings: bool = Field(
        True, description="Use precomputed embeddings from L2 candidates if available"
    )
    cache_embeddings: bool = Field(False, description="Cache computed embeddings back to L2")
    max_length: int = Field(
        None,
        description="Maximum sequence length for tokenization. Passed to GLiNER.from_pretrained.",
    )


# TODO replace candidates with labels
class L3Input(BaseInput):
    texts: list[str] = Field(...)
    labels: list[list[Any]] = Field(...)


class L3Entity(BaseOutput):
    text: str
    label: str
    start: int
    end: int
    score: float
    class_probs: dict[str, float] | None = Field(
        None, description="Per-label class probabilities from GLiNER"
    )


class L3Output(BaseOutput):
    entities: list[list[L3Entity]] = Field(...)
