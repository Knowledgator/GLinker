from pydantic import Field
from typing import Any
from src.core.base import BaseConfig, BaseInput, BaseOutput


class L0Config(BaseConfig):
    """Base config for converters"""
    pass


class L1ToL2Config(L0Config):
    """Config for L1 to L2 converter"""
    deduplicate_mentions: bool = Field(True, description="Deduplicate extracted mentions")
    min_mention_length: int = Field(2, description="Minimum mention length")


class L2ToL3Config(L0Config):
    """Config for L2 to L3 converter"""
    deduplicate_candidates: bool = Field(True, description="Deduplicate candidates by entity_id")
    flatten_per_text: bool = Field(True, description="Flatten candidates per text")