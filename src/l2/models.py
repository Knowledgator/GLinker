from pydantic import Field
from typing import List, Union
from src.core.base import BaseConfig, BaseInput, BaseOutput


class PostgresConfig(BaseConfig):
    host: str = Field("localhost")
    port: int = Field(5432)
    database: str = Field(...)
    user: str = Field(...)
    password: str = Field(...)


class ElasticsearchConfig(BaseConfig):
    hosts: List[str] = Field(["localhost:9200"])
    index_name: str = Field("entities")
    api_key: str = Field(None)


class RedisConfig(BaseConfig):
    host: str = Field("localhost")
    port: int = Field(6379)
    db: int = Field(0)
    password: str = Field(None)


class L2Config(BaseConfig):
    database_type: str = Field(..., description="Type: postgres, elasticsearch, redis")
    database_config: Union[PostgresConfig, ElasticsearchConfig, RedisConfig, dict] = Field(...)
    max_candidates: int = Field(30)
    fuzzy_max_distance: int = Field(2)
    use_fuzzy: bool = Field(True)
    min_popularity: int = Field(0)


class L2Input(BaseInput):
    mentions: List[str] = Field(..., description="List of mentions to search")


class L2Candidate(BaseOutput):
    entity_id: str
    label: str
    entity_type: str
    popularity: int
    aliases: List[str]


class L2Output(BaseOutput):
    candidates: List[List[L2Candidate]] = Field(...)