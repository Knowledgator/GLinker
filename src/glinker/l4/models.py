from pydantic import Field
from typing import List, Any, Optional
from glinker.core.base import BaseConfig, BaseInput, BaseOutput


class L4Config(BaseConfig):
    model_name: str = Field(...)
    token: str = Field(None)
    device: str = Field("cpu")
    threshold: float = Field(0.5)
    flat_ner: bool = Field(True)
    multi_label: bool = Field(False)
    max_labels: int = Field(
        20,
        description="Maximum number of candidate labels per inference call. "
        "When candidates exceed this, they are split into chunks."
    )
    max_length: int = Field(
        None,
        description="Maximum sequence length for tokenization. Passed to GLiNER.from_pretrained."
    )
