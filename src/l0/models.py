from pydantic import Field
from typing import List, Optional
from src.core.base import BaseConfig, BaseInput, BaseOutput
from src.l1.models import L1Entity
from src.l2.models import DatabaseRecord
from src.l3.models import L3Entity


class L0Config(BaseConfig):
    """L0 aggregation configuration"""
    min_confidence: float = Field(0.0, description="Minimum confidence threshold for linked entities")
    include_unlinked: bool = Field(True, description="Include mentions without linked entities")
    return_all_candidates: bool = Field(False, description="Return all candidates or only top match")


class L0Input(BaseInput):
    """L0 processor input - outputs from L1, L2, L3"""
    l1_entities: List[List[L1Entity]] = Field(..., description="Entities from L1 (per text)")
    l2_candidates: List[List[DatabaseRecord]] = Field(..., description="Candidates from L2 (per mention)")
    l3_entities: List[List[L3Entity]] = Field(..., description="Linked entities from L3 (per text)")


class LinkedEntity(BaseOutput):
    """Linked entity information from L3"""
    entity_id: str = Field(..., description="Entity ID from matched candidate")
    label: str = Field(..., description="Entity label")
    confidence: float = Field(..., description="Linking confidence score from L3")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    matched_text: str = Field(..., description="Matched text from L3")


class L0Entity(BaseOutput):
    """
    Aggregated entity combining information from all layers:
    - L1: mention detection (text, position, context)
    - L2: candidates (entity database records)
    - L3: disambiguation (linked entity with confidence)
    """
    # From L1 - mention detection
    mention_text: str = Field(..., description="Extracted mention text from L1")
    mention_start: int = Field(..., description="Start position in original text")
    mention_end: int = Field(..., description="End position in original text")
    left_context: str = Field(..., description="Left context from L1")
    right_context: str = Field(..., description="Right context from L1")

    # From L2 - candidate retrieval
    candidates: List[DatabaseRecord] = Field(
        default_factory=list,
        description="All candidates found in L2 for this mention"
    )
    num_candidates: int = Field(0, description="Number of candidates found")

    # From L3 - entity linking
    linked_entity: Optional[LinkedEntity] = Field(
        None,
        description="Linked entity if disambiguation was successful"
    )
    is_linked: bool = Field(False, description="Whether entity was successfully linked")

    # Aggregated metadata
    pipeline_stage: str = Field(
        "",
        description="Last successful stage: 'l1_only', 'l2_found', 'l3_linked'"
    )


class L0Output(BaseOutput):
    """L0 processor output"""
    entities: List[List[L0Entity]] = Field(
        ...,
        description="Aggregated entities per text with full pipeline information"
    )
    stats: dict = Field(
        default_factory=dict,
        description="Pipeline statistics (total, linked, unlinked, etc.)"
    )
