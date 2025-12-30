from pydantic import Field
from typing import List, Dict, Any, Optional, Literal
from src.core.base import BaseConfig, BaseInput, BaseOutput
from src.core.database_record import DatabaseRecord


class FuzzyConfig(BaseConfig):
    max_distance: int = Field(2)
    min_similarity: float = Field(0.3)
    n_gram_size: int = Field(3)
    prefix_length: int = Field(1)


class LayerConfig(BaseConfig):
    type: str = Field(...)
    priority: int = Field(...)
    config: Dict[str, Any] = Field(...)
    
    search_mode: List[Literal["exact", "fuzzy"]] = Field(
        ["exact"],
        description="Search methods: ['exact'], ['fuzzy'], or ['exact', 'fuzzy']"
    )
    
    write: bool = Field(True)
    cache_policy: str = Field("always")
    ttl: int = Field(3600)
    field_mapping: Dict[str, str] = Field(...)
    fuzzy: Optional[FuzzyConfig] = Field(default_factory=FuzzyConfig)


class L2Config(BaseConfig):
    layers: List[LayerConfig] = Field(...)
    max_candidates: int = Field(30)
    min_popularity: int = Field(0)


class L2Input(BaseInput):
    mentions: List[str] = Field(...)
    structure: List[List[str]] = Field(None)


class L2Output(BaseOutput):
    candidates: List[List[DatabaseRecord]] = Field(...)