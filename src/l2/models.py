from pydantic import Field, BaseModel
from typing import List, Dict, Any, Optional, Literal
from src.core.base import BaseConfig, BaseInput, BaseOutput


class DatabaseRecord(BaseModel):
    """
    Unified format for all database layers
    
    All layers (Redis, Elasticsearch, Postgres) map their fields to this format.
    This ensures consistency across different data sources.
    """
    entity_id: str = Field(..., description="Unique entity identifier")
    label: str = Field(..., description="Primary label/name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    description: str = Field(default="", description="Entity description")
    entity_type: str = Field(default="", description="Entity type/category")
    popularity: int = Field(default=0, description="Popularity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Database-specific metadata"
    )
    source: str = Field(default="", description="Source layer: redis|elasticsearch|postgres")


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