from pydantic import BaseModel, Field
from typing import List, Dict, Any


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