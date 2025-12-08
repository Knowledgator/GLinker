from abc import abstractmethod
from typing import List
import psycopg2
import redis
from elasticsearch import Elasticsearch
from src.core.base import BaseComponent
from src.core.registry import component_registry
from .models import L2Config, L2Candidate, PostgresConfig, ElasticsearchConfig, RedisConfig


class L2BaseComponent(BaseComponent[L2Config]):
    """Base for all L2 components - only common utilities"""
    
    @abstractmethod
    def get_available_methods(self) -> List[str]:
        pass
    
    def filter_by_popularity(
        self, 
        candidates: List[L2Candidate],
        min_popularity: int = None
    ) -> List[L2Candidate]:
        threshold = min_popularity if min_popularity is not None else self.config.min_popularity
        return [c for c in candidates if c.popularity >= threshold]
    
    def deduplicate_candidates(self, candidates: List[L2Candidate]) -> List[L2Candidate]:
        seen = set()
        unique = []
        for candidate in candidates:
            if candidate.entity_id not in seen:
                unique.append(candidate)
                seen.add(candidate.entity_id)
        return unique
    
    def limit_candidates(
        self, 
        candidates: List[L2Candidate],
        limit: int = None
    ) -> List[L2Candidate]:
        max_limit = limit if limit is not None else self.config.max_candidates
        return candidates[:max_limit]


@component_registry.register("l2_postgres")
class L2PostgresComponent(L2BaseComponent):
    """PostgreSQL component with its specific methods"""
    
    def _setup(self):
        db_cfg = self.config.database_config
        if isinstance(db_cfg, dict):
            db_cfg = PostgresConfig(**db_cfg)
        
        self.conn = psycopg2.connect(
            host=db_cfg.host,
            port=db_cfg.port,
            database=db_cfg.database,
            user=db_cfg.user,
            password=db_cfg.password
        )
    
    def get_available_methods(self) -> List[str]:
        return [
            "search_exact",
            "search_fuzzy_levenshtein",
            "search_by_popularity",
            "search_by_type",
            "bulk_search_exact",
            "filter_by_popularity",
            "deduplicate_candidates",
            "limit_candidates"
        ]
    
    def search_exact(self, mention: str, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT e.entity_id, e.label, e.entity_type, e.popularity,
                   ARRAY_AGG(DISTINCT a.alias_text) as aliases
            FROM entities e
            JOIN aliases a ON e.entity_id = a.entity_id
            WHERE LOWER(a.alias_text) = LOWER(%s)
            GROUP BY e.entity_id, e.label, e.entity_type, e.popularity
            ORDER BY e.popularity DESC
            LIMIT %s
        """, (mention, max_limit))
        
        candidates = []
        for row in cursor.fetchall():
            candidates.append(L2Candidate(
                entity_id=row[0],
                label=row[1],
                entity_type=row[2],
                popularity=row[3],
                aliases=row[4] or []
            ))
        
        cursor.close()
        return candidates
    
    def search_fuzzy_levenshtein(self, mention: str, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT e.entity_id, e.label, e.entity_type, e.popularity,
                   ARRAY_AGG(DISTINCT a.alias_text) as aliases,
                   MIN(levenshtein(LOWER(a.alias_text), LOWER(%s))) as distance
            FROM entities e
            JOIN aliases a ON e.entity_id = a.entity_id
            WHERE levenshtein(LOWER(a.alias_text), LOWER(%s)) <= %s
            GROUP BY e.entity_id, e.label, e.entity_type, e.popularity
            ORDER BY distance ASC, e.popularity DESC
            LIMIT %s
        """, (mention, mention, self.config.fuzzy_max_distance, max_limit))
        
        candidates = []
        for row in cursor.fetchall():
            candidates.append(L2Candidate(
                entity_id=row[0],
                label=row[1],
                entity_type=row[2],
                popularity=row[3],
                aliases=row[4] or []
            ))
        
        cursor.close()
        return candidates
    
    def search_by_popularity(self, mention: str, min_pop: int = 10, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT e.entity_id, e.label, e.entity_type, e.popularity,
                   ARRAY_AGG(DISTINCT a.alias_text) as aliases
            FROM entities e
            JOIN aliases a ON e.entity_id = a.entity_id
            WHERE LOWER(a.alias_text) = LOWER(%s) AND e.popularity >= %s
            GROUP BY e.entity_id, e.label, e.entity_type, e.popularity
            ORDER BY e.popularity DESC
            LIMIT %s
        """, (mention, min_pop, max_limit))
        
        candidates = []
        for row in cursor.fetchall():
            candidates.append(L2Candidate(
                entity_id=row[0],
                label=row[1],
                entity_type=row[2],
                popularity=row[3],
                aliases=row[4] or []
            ))
        
        cursor.close()
        return candidates
    
    def search_by_type(self, mention: str, entity_type: str, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT e.entity_id, e.label, e.entity_type, e.popularity,
                   ARRAY_AGG(DISTINCT a.alias_text) as aliases
            FROM entities e
            JOIN aliases a ON e.entity_id = a.entity_id
            WHERE LOWER(a.alias_text) = LOWER(%s) AND e.entity_type = %s
            GROUP BY e.entity_id, e.label, e.entity_type, e.popularity
            ORDER BY e.popularity DESC
            LIMIT %s
        """, (mention, entity_type, max_limit))
        
        candidates = []
        for row in cursor.fetchall():
            candidates.append(L2Candidate(
                entity_id=row[0],
                label=row[1],
                entity_type=row[2],
                popularity=row[3],
                aliases=row[4] or []
            ))
        
        cursor.close()
        return candidates
    
    def bulk_search_exact(self, mentions: List[str], limit: int = None) -> List[List[L2Candidate]]:
        max_limit = limit or self.config.max_candidates
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT a.alias_text, e.entity_id, e.label, e.entity_type, e.popularity,
                   ARRAY_AGG(DISTINCT a2.alias_text) as aliases
            FROM unnest(%s::text[]) AS input_alias(alias_text)
            JOIN aliases a ON LOWER(a.alias_text) = LOWER(input_alias.alias_text)
            JOIN entities e ON e.entity_id = a.entity_id
            JOIN aliases a2 ON e.entity_id = a2.entity_id
            GROUP BY a.alias_text, e.entity_id, e.label, e.entity_type, e.popularity
            ORDER BY a.alias_text, e.popularity DESC
        """, (mentions,))
        
        results = {}
        for row in cursor.fetchall():
            mention = row[0]
            if mention not in results:
                results[mention] = []
            
            if len(results[mention]) < max_limit:
                results[mention].append(L2Candidate(
                    entity_id=row[1],
                    label=row[2],
                    entity_type=row[3],
                    popularity=row[4],
                    aliases=row[5] or []
                ))
        
        cursor.close()
        return [results.get(m, []) for m in mentions]
    
    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


@component_registry.register("l2_elasticsearch")
class L2ElasticsearchComponent(L2BaseComponent):
    """Elasticsearch component with its specific methods"""
    
    def _setup(self):
        es_cfg = self.config.database_config
        if isinstance(es_cfg, dict):
            es_cfg = ElasticsearchConfig(**es_cfg)
        
        self.index_name = es_cfg.index_name
        
        if es_cfg.api_key:
            self.client = Elasticsearch(es_cfg.hosts, api_key=es_cfg.api_key)
        else:
            self.client = Elasticsearch(es_cfg.hosts)
    
    def get_available_methods(self) -> List[str]:
        return [
            "search_exact",
            "search_fuzzy_match",
            "search_semantic",
            "search_multi_field",
            "search_with_boost",
            "aggregation_by_type",
            "filter_by_popularity",
            "deduplicate_candidates",
            "limit_candidates"
        ]
    
    def search_exact(self, mention: str, limit: int = None) -> List[L2Candidate]:
        """Elasticsearch exact term search"""
        max_limit = limit or self.config.max_candidates
        
        # USE MATCH INSTEAD OF TERM for case-insensitive search
        query = {
            "query": {
                "match": {
                    "aliases": {
                        "query": mention,
                        "operator": "and"
                    }
                }
            },
            "size": max_limit,
            "sort": [{"popularity": {"order": "desc"}}]
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        candidates = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            candidates.append(L2Candidate(
                entity_id=source['entity_id'],
                label=source['label'],
                entity_type=source['entity_type'],
                popularity=source['popularity'],
                aliases=source.get('aliases', [])
            ))
        
        return candidates
    
    def search_fuzzy_match(self, mention: str, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        
        query = {
            "query": {
                "match": {
                    "aliases": {
                        "query": mention,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "size": max_limit,
            "sort": [
                {"_score": {"order": "desc"}},
                {"popularity": {"order": "desc"}}
            ]
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        candidates = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            candidates.append(L2Candidate(
                entity_id=source['entity_id'],
                label=source['label'],
                entity_type=source['entity_type'],
                popularity=source['popularity'],
                aliases=source.get('aliases', [])
            ))
        
        return candidates
    
    def search_semantic(self, mention: str, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        
        query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": self._get_embedding(mention)}
                    }
                }
            },
            "size": max_limit
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        candidates = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            candidates.append(L2Candidate(
                entity_id=source['entity_id'],
                label=source['label'],
                entity_type=source['entity_type'],
                popularity=source['popularity'],
                aliases=source.get('aliases', [])
            ))
        
        return candidates
    
    def search_multi_field(self, mention: str, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        
        query = {
            "query": {
                "multi_match": {
                    "query": mention,
                    "fields": ["label^3", "aliases^2", "description"],
                    "type": "best_fields"
                }
            },
            "size": max_limit,
            "sort": [
                {"_score": {"order": "desc"}},
                {"popularity": {"order": "desc"}}
            ]
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        candidates = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            candidates.append(L2Candidate(
                entity_id=source['entity_id'],
                label=source['label'],
                entity_type=source['entity_type'],
                popularity=source['popularity'],
                aliases=source.get('aliases', [])
            ))
        
        return candidates
    
    def search_with_boost(self, mention: str, boost_popular: bool = True, limit: int = None) -> List[L2Candidate]:
        max_limit = limit or self.config.max_candidates
        
        query = {
            "query": {
                "function_score": {
                    "query": {"match": {"aliases": mention}},
                    "field_value_factor": {
                        "field": "popularity",
                        "modifier": "log1p",
                        "factor": 1.2
                    }
                }
            },
            "size": max_limit
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        candidates = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            candidates.append(L2Candidate(
                entity_id=source['entity_id'],
                label=source['label'],
                entity_type=source['entity_type'],
                popularity=source['popularity'],
                aliases=source.get('aliases', [])
            ))
        
        return candidates
    
    def aggregation_by_type(self, mention: str) -> dict:
        query = {
            "query": {"match": {"aliases": mention}},
            "size": 0,
            "aggs": {
                "types": {
                    "terms": {
                        "field": "entity_type.keyword",
                        "size": 20
                    }
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=query)
        return response['aggregations']['types']['buckets']
    
    def _get_embedding(self, text: str) -> List[float]:
        return [0.0] * 768
    
    def __del__(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

@component_registry.register("l2_redis")
class L2RedisComponent(L2BaseComponent):
    """Redis component with its specific methods"""
    
    def _setup(self):
        redis_cfg = self.config.database_config
        if isinstance(redis_cfg, dict):
            redis_cfg = RedisConfig(**redis_cfg)
        
        self.client = redis.Redis(
            host=redis_cfg.host,
            port=redis_cfg.port,
            db=redis_cfg.db,
            password=redis_cfg.password,
            decode_responses=True
        )
    
    def get_available_methods(self) -> List[str]:
        return [
            "search_exact",
            "search_by_prefix",
            "search_by_popularity",
            "search_fuzzy_scan",
            "get_by_id",
            "filter_by_popularity",
            "deduplicate_candidates",
            "limit_candidates"
        ]
    
    def search_exact(self, mention: str, limit: int = None) -> List[L2Candidate]:
        """Search for exact matches in Redis"""
        max_limit = limit or self.config.max_candidates
        
        # Get entity IDs from alias
        alias_key = f"alias:{mention.lower()}"
        entity_ids = self.client.smembers(alias_key)
        
        if not entity_ids:
            return []
        
        # Get entity data and sort by popularity
        candidates = []
        for entity_id in entity_ids:
            entity_data = self.client.hgetall(f"entity:{entity_id}")
            if entity_data:
                aliases = self.client.smembers(f"entity:{entity_id}:aliases")
                candidates.append(L2Candidate(
                    entity_id=entity_data['entity_id'],
                    label=entity_data['label'],
                    entity_type=entity_data['entity_type'],
                    popularity=int(entity_data['popularity']),
                    aliases=list(aliases)
                ))
        
        # Sort by popularity
        candidates.sort(key=lambda x: x.popularity, reverse=True)
        return candidates[:max_limit]
    
    def search_by_prefix(self, prefix: str, limit: int = None) -> List[L2Candidate]:
        """Redis-specific: search by prefix using SCAN"""
        max_limit = limit or self.config.max_candidates
        
        pattern = f"alias:{prefix.lower()}*"
        entity_ids = set()
        
        for key in self.client.scan_iter(match=pattern, count=100):
            ids = self.client.smembers(key)
            entity_ids.update(ids)
            if len(entity_ids) >= max_limit * 2:
                break
        
        candidates = []
        for entity_id in entity_ids:
            entity_data = self.client.hgetall(f"entity:{entity_id}")
            if entity_data:
                aliases = self.client.smembers(f"entity:{entity_id}:aliases")
                candidates.append(L2Candidate(
                    entity_id=entity_data['entity_id'],
                    label=entity_data['label'],
                    entity_type=entity_data['entity_type'],
                    popularity=int(entity_data['popularity']),
                    aliases=list(aliases)
                ))
        
        candidates.sort(key=lambda x: x.popularity, reverse=True)
        return candidates[:max_limit]
    
    def search_by_popularity(self, mention: str, min_pop: int = 10, limit: int = None) -> List[L2Candidate]:
        """Redis-specific: search with popularity threshold"""
        candidates = self.search_exact(mention, limit)
        return [c for c in candidates if c.popularity >= min_pop]
    
    def search_fuzzy_scan(self, mention: str, limit: int = None) -> List[L2Candidate]:
        """Redis-specific: fuzzy search using Levenshtein on scanned aliases"""
        from difflib import SequenceMatcher
        
        max_limit = limit or self.config.max_candidates
        mention_lower = mention.lower()
        
        # Scan all aliases
        candidates_map = {}
        for key in self.client.scan_iter(match="alias:*", count=1000):
            alias = key.replace("alias:", "")
            
            # Simple fuzzy matching
            similarity = SequenceMatcher(None, mention_lower, alias).ratio()
            if similarity > 0.7:  # 70% similarity threshold
                entity_ids = self.client.smembers(key)
                for entity_id in entity_ids:
                    if entity_id not in candidates_map:
                        entity_data = self.client.hgetall(f"entity:{entity_id}")
                        if entity_data:
                            aliases = self.client.smembers(f"entity:{entity_id}:aliases")
                            candidates_map[entity_id] = L2Candidate(
                                entity_id=entity_data['entity_id'],
                                label=entity_data['label'],
                                entity_type=entity_data['entity_type'],
                                popularity=int(entity_data['popularity']),
                                aliases=list(aliases)
                            )
        
        candidates = list(candidates_map.values())
        candidates.sort(key=lambda x: x.popularity, reverse=True)
        return candidates[:max_limit]
    
    def get_by_id(self, entity_id: str) -> L2Candidate:
        """Redis-specific: get entity by ID"""
        entity_data = self.client.hgetall(f"entity:{entity_id}")
        if not entity_data:
            return None
        
        aliases = self.client.smembers(f"entity:{entity_id}:aliases")
        return L2Candidate(
            entity_id=entity_data['entity_id'],
            label=entity_data['label'],
            entity_type=entity_data['entity_type'],
            popularity=int(entity_data['popularity']),
            aliases=list(aliases)
        )
    
    def __del__(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()