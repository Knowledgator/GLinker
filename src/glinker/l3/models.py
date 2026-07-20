from typing import Any, Dict, List

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
    span_extraction_threshold: float | None = Field(
        None,
        description=(
            "Threshold for BIO span extraction inside model.forward. "
            "Lower → more candidate spans constructed → higher recall. "
            "When None, falls back to `threshold`. "
            "Useful to set low (0.1–0.2) so final `threshold` filters span_logits "
            "without also cutting span candidates."
        ),
    )
    max_length: int = Field(
        None,
        description="Maximum sequence length for text tokenization. Passed to GLiNER.from_pretrained.",
    )
    label_max_length: int = Field(
        None,
        description=(
            "Maximum token length for label encoding. "
            "Overrides max_length for the labels tokenizer only. "
            "Useful when labels are short descriptions and you want to save memory."
        ),
    )


class L3LLMConfig(BaseConfig):
    provider: str = Field("gemini", description="LLM provider: 'gemini' or 'openai'")
    model_name: str = Field("gemini-2.0-flash")
    api_key: str | None = Field(None, description="API key (falls back to env var)")
    top_k: int = Field(5, description="Max candidates shown to the LLM per mention")
    threshold: float = Field(0.0)
    batch_size: int = Field(10, description="Mentions per LLM request")
    max_retries: int = Field(3)
    retry_delay: float = Field(1.0)
    temperature: float = Field(0.0)


# TODO replace candidates with labels
class L3Input(BaseInput):
    texts: List[str] = Field(...)
    labels: List[List[Any]] = Field(...)


class L3Entity(BaseOutput):
    text: str
    label: str
    start: int
    end: int
    score: float
    class_probs: Dict[str, float] | None = Field(
        None, description="Per-label class probabilities from GLiNER"
    )


class L3Output(BaseOutput):
    entities: List[List[L3Entity]] = Field(...)
