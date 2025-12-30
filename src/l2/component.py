from abc import ABC, abstractmethod
from typing import List, Dict, Any
import redis
import json
from elasticsearch import Elasticsearch
import psycopg2
from psycopg2.extras import RealDictCursor

from src.core.base import BaseComponent
from .models import L2Config, LayerConfig, FuzzyConfig, DatabaseRecord


class DatabaseLayer(ABC):
    def __init__(self, config: LayerConfig):
        self.config = config
        self.priority = config.priority
        self.ttl = config.ttl
        self.write = config.write
        self.cache_policy = config.cache_policy
        self.field_mapping = config.field_mapping
        self.fuzzy_config = config.fuzzy or FuzzyConfig()
        self._setup()
    
    @abstractmethod
    def _setup(self):
        pass
    
    def normalize_query(self, query: str) -> str:
        return query.lower().strip()
    
    @abstractmethod
    def search(self, query: str) -> List[DatabaseRecord]:
        pass
    
    @abstractmethod
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        pass
    
    def supports_fuzzy(self) -> bool:
        return self.fuzzy_config is not None
    
    @abstractmethod
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    def map_to_record(self, raw_data: Dict[str, Any]) -> DatabaseRecord:
        mapped = {}
        for standard_field, db_field in self.field_mapping.items():
            if db_field in raw_data:
                mapped[standard_field] = raw_data[db_field]
        mapped['source'] = self.config.type
        return DatabaseRecord(**mapped)


class RedisLayer(DatabaseLayer):
    def _setup(self):
        self.client = redis.Redis(
            host=self.config.config.get('host', 'localhost'),
            port=self.config.config.get('port', 6379),
            db=self.config.config.get('db', 0),
            password=self.config.config.get('password'),
            decode_responses=False
        )
    
    def supports_fuzzy(self) -> bool:
        return False
    
    def search(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        key = f"entity:{query}"
        
        try:
            data = self.client.get(key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                
                records_data = json.loads(data)
                
                if isinstance(records_data, list):
                    results = []
                    for r in records_data:
                        if isinstance(r, dict):
                            r['source'] = 'redis'
                            results.append(DatabaseRecord(**r))
                        else:
                            results.append(r)
                    return results
                
                elif isinstance(records_data, dict):
                    records_data['source'] = 'redis'
                    return [DatabaseRecord(**records_data)]
        
        except Exception as e:
            print(f"[ERROR Redis] Search error: {e}")
        
        return []
    
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        return []
    
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        key = self.normalize_query(key)
        cache_key = f"entity:{key}"
        
        try:
            data = json.dumps([r.dict() for r in records])
            self.client.setex(cache_key, ttl, data)
        except Exception as e:
            print(f"[ERROR Redis] Write error: {e}")
    
    def is_available(self) -> bool:
        try:
            self.client.ping()
            return True
        except:
            return False


class ElasticsearchLayer(DatabaseLayer):
    def _setup(self):
        self.client = Elasticsearch(
            self.config.config['hosts'],
            api_key=self.config.config.get('api_key')
        )
        self.index_name = self.config.config['index_name']
    
    def search(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        
        try:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["label^2", "aliases^1.5", "description"],
                        "type": "best_fields"
                    }
                },
                "size": 50
            }
            response = self.client.search(index=self.index_name, body=body)
            return self._process_hits(response['hits']['hits'])
        except Exception as e:
            print(f"[ERROR ES] Search error: {e}")
            return []
    
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        fuzzy_distance = self.fuzzy_config.max_distance
        
        try:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["label^2", "aliases^1.5", "description"],
                        "fuzziness": fuzzy_distance,
                        "prefix_length": self.fuzzy_config.prefix_length,
                        "max_expansions": 50
                    }
                },
                "size": 50
            }
            response = self.client.search(index=self.index_name, body=body)
            return self._process_hits(response['hits']['hits'])
        except Exception as e:
            print(f"[ERROR ES] Fuzzy error: {e}")
            return []
    
    def _process_hits(self, hits: List[Dict]) -> List[DatabaseRecord]:
        records = []
        for hit in hits:
            source = hit['_source']
            source['_id'] = hit['_id']
            source['source'] = 'elasticsearch'
            record = self.map_to_record(source)
            records.append(record)
        return records
    
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        if not records:
            return
        
        try:
            bulk_body = []
            for record in records:
                bulk_body.append({"index": {"_index": self.index_name, "_id": record.entity_id}})
                bulk_body.append(self._map_from_record(record))
            
            if bulk_body:
                self.client.bulk(body=bulk_body, refresh=True)
        except Exception as e:
            print(f"[ERROR ES] Write error: {e}")
    
    def _map_from_record(self, record: DatabaseRecord) -> dict:
        reverse_mapping = {v: k for k, v in self.field_mapping.items()}
        doc = {}
        for standard_field, value in record.dict().items():
            if standard_field == 'source':
                continue
            if standard_field in reverse_mapping:
                doc[reverse_mapping[standard_field]] = value
            else:
                doc[standard_field] = value
        return doc
    
    def is_available(self) -> bool:
        try:
            return self.client.ping()
        except:
            return False


class PostgresLayer(DatabaseLayer):
    def _setup(self):
        self.conn = psycopg2.connect(
            host=self.config.config['host'],
            port=self.config.config.get('port', 5432),
            database=self.config.config['database'],
            user=self.config.config['user'],
            password=self.config.config['password']
        )
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            self.conn.commit()
        except Exception as e:
            print(f"[WARN Postgres] pg_trgm: {e}")
        finally:
            cursor.close()
    
    def search(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            sql = """
                SELECT 
                    e.entity_id,
                    e.label,
                    e.description,
                    e.entity_type,
                    e.popularity,
                    COALESCE(array_agg(a.alias) FILTER (WHERE a.alias IS NOT NULL), ARRAY[]::text[]) as aliases
                FROM entities e
                LEFT JOIN aliases a ON e.entity_id = a.entity_id
                WHERE LOWER(e.label) LIKE %s
                   OR EXISTS (
                       SELECT 1 FROM aliases a2 
                       WHERE a2.entity_id = e.entity_id 
                       AND LOWER(a2.alias) LIKE %s
                   )
                GROUP BY e.entity_id, e.label, e.description, e.entity_type, e.popularity
                ORDER BY e.popularity DESC
                LIMIT 50
            """
            cursor.execute(sql, (f"%{query}%", f"%{query}%"))
            records = self._process_rows(cursor.fetchall())
            cursor.close()
            return records
        except Exception as e:
            print(f"[ERROR Postgres] Search error: {e}")
            return []
    
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        threshold = self.fuzzy_config.min_similarity
        
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            sql = """
                SELECT 
                    e.entity_id,
                    e.label,
                    e.description,
                    e.entity_type,
                    e.popularity,
                    COALESCE(array_agg(a.alias) FILTER (WHERE a.alias IS NOT NULL), ARRAY[]::text[]) as aliases,
                    similarity(LOWER(e.label), %s) AS sim_score
                FROM entities e
                LEFT JOIN aliases a ON e.entity_id = a.entity_id
                WHERE similarity(LOWER(e.label), %s) >= %s
                GROUP BY e.entity_id, e.label, e.description, e.entity_type, e.popularity
                ORDER BY sim_score DESC, e.popularity DESC
                LIMIT 50
            """
            cursor.execute(sql, (query, query, threshold))
            records = self._process_rows(cursor.fetchall())
            cursor.close()
            return records
        except Exception as e:
            print(f"[ERROR Postgres] Fuzzy error: {e}")
            return self.search(query)
    
    def _process_rows(self, rows: List[Dict]) -> List[DatabaseRecord]:
        records = []
        for row in rows:
            row_dict = dict(row)
            row_dict['source'] = 'postgres'
            record = self.map_to_record(row_dict)
            records.append(record)
        return records
    
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        pass
    
    def is_available(self) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except:
            return False


class DatabaseChainComponent(BaseComponent[L2Config]):
    """Multi-layer database chain component"""
    
    def _setup(self):
        self.layers: List[DatabaseLayer] = []
        
        for layer_config in self.config.layers:
            if isinstance(layer_config, dict):
                layer_config = LayerConfig(**layer_config)
            
            if layer_config.type == "redis":
                layer = RedisLayer(layer_config)
            elif layer_config.type == "elasticsearch":
                layer = ElasticsearchLayer(layer_config)
            elif layer_config.type == "postgres":
                layer = PostgresLayer(layer_config)
            else:
                raise ValueError(f"Unknown layer type: {layer_config.type}")
            
            self.layers.append(layer)
        
        self.layers.sort(key=lambda x: x.priority)
    
    def get_available_methods(self) -> List[str]:
        return [
            "search",
            "filter_by_popularity",
            "deduplicate_candidates",
            "limit_candidates",
            "sort_by_popularity"
        ]
    
    def search(self, mention: str) -> List[DatabaseRecord]:
        found_in_layer = None
        results = []
        
        for layer in self.layers:
            if not layer.is_available():
                continue
            
            layer_results = []
            
            for mode in layer.config.search_mode:
                if mode == "exact":
                    layer_results.extend(layer.search(mention))
                elif mode == "fuzzy":
                    if layer.supports_fuzzy():
                        layer_results.extend(layer.search_fuzzy(mention))
            
            if layer_results:
                layer_results = self.deduplicate_candidates(layer_results)
                results = layer_results
                found_in_layer = layer
                break
        
        if results and found_in_layer:
            self._cache_write(mention, results, found_in_layer)
        
        return results
    
    def _cache_write(self, query: str, results: List[DatabaseRecord], source_layer: DatabaseLayer):
        for layer in self.layers:
            if layer.priority >= source_layer.priority:
                continue
            if not layer.write:
                continue
            
            if layer.cache_policy == "always":
                layer.write_cache(query, results, layer.ttl)
            elif layer.cache_policy == "miss":
                existing = layer.search(query)
                if not existing:
                    layer.write_cache(query, results, layer.ttl)
            elif layer.cache_policy == "hit":
                existing = layer.search(query)
                if existing:
                    layer.write_cache(query, results, layer.ttl)
    
    def filter_by_popularity(self, records: List[DatabaseRecord], min_popularity: int = None) -> List[DatabaseRecord]:
        threshold = min_popularity if min_popularity is not None else self.config.min_popularity
        return [r for r in records if r.popularity >= threshold]
    
    def deduplicate_candidates(self, records: List[DatabaseRecord]) -> List[DatabaseRecord]:
        seen = set()
        unique = []
        for record in records:
            if record.entity_id not in seen:
                unique.append(record)
                seen.add(record.entity_id)
        return unique
    
    def limit_candidates(self, records: List[DatabaseRecord], limit: int = None) -> List[DatabaseRecord]:
        max_cands = limit if limit is not None else self.config.max_candidates
        return records[:max_cands]
    
    def sort_by_popularity(self, records: List[DatabaseRecord]) -> List[DatabaseRecord]:
        return sorted(records, key=lambda x: x.popularity, reverse=True)