from pydantic import Field, BaseModel
from typing import List, Dict, Any, Optional, Literal
from src.core.base import BaseConfig, BaseInput, BaseOutput


class DatabaseRecord(BaseModel):
    """
    Unified format for all database layers
    
    All layers (Dict, Redis, Elasticsearch, Postgres) use this format.
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
    source: str = Field(default="", description="Source layer: dict|redis|elasticsearch|postgres")


class FuzzyConfig(BaseConfig):
    """Fuzzy search configuration"""
    max_distance: int = Field(2, description="Maximum Levenshtein distance")
    min_similarity: float = Field(0.3, description="Minimum similarity threshold")
    n_gram_size: int = Field(3, description="N-gram size for matching")
    prefix_length: int = Field(1, description="Prefix length to preserve")


class LayerConfig(BaseConfig):
    """Database layer configuration"""
    type: str = Field(..., description="Layer type: dict|redis|elasticsearch|postgres")
    priority: int = Field(..., description="Search priority (0 = highest)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Layer-specific config")
    
    search_mode: List[Literal["exact", "fuzzy"]] = Field(
        ["exact"],
        description="Search methods: ['exact'], ['fuzzy'], or ['exact', 'fuzzy']"
    )
    
    write: bool = Field(True, description="Enable write operations")
    cache_policy: str = Field("always", description="Cache policy: always|miss|hit")
    ttl: int = Field(3600, description="TTL in seconds (0 = no expiry)")
    field_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "entity_id": "entity_id",
            "label": "label",
            "aliases": "aliases",
            "description": "description",
            "entity_type": "entity_type",
            "popularity": "popularity"
        },
        description="Field mapping: DatabaseRecord field -> storage field"
    )
    fuzzy: Optional[FuzzyConfig] = Field(default_factory=FuzzyConfig, description="Fuzzy search config")


class L2Config(BaseConfig):
    """L2 processor configuration"""
    layers: List[LayerConfig] = Field(..., description="Database layers in priority order")
    max_candidates: int = Field(30, description="Maximum candidates per mention")
    min_popularity: int = Field(0, description="Minimum popularity threshold")


class L2Input(BaseInput):
    """L2 processor input"""
    mentions: List[str] = Field(..., description="List of mentions to search")
    structure: List[List[str]] = Field(None, description="Optional grouping structure")


class L2Output(BaseOutput):
    """L2 processor output"""
    candidates: List[List[DatabaseRecord]] = Field(..., description="Candidates per mention/group")