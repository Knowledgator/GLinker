from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set
import redis
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

from glinker.core.base import BaseComponent
from .models import L2Config, LayerConfig, FuzzyConfig, DatabaseRecord


class DatabaseLayer(ABC):
    """Base class for all database layers"""
    
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
        """Initialize layer resources"""
        pass
    
    def normalize_query(self, query: str) -> str:
        """Normalize query for search"""
        return query.lower().strip()
    
    @abstractmethod
    def search(self, query: str) -> List[DatabaseRecord]:
        """Exact search"""
        pass
    
    @abstractmethod
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Fuzzy search"""
        pass
    
    def supports_fuzzy(self) -> bool:
        """Check if layer supports fuzzy search"""
        return self.fuzzy_config is not None
    
    @abstractmethod
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        """Write records to cache"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if layer is available"""
        pass
    
    @abstractmethod
    def load_bulk(self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000) -> int:
        """Bulk load entities"""
        pass
    
    def clear(self):
        """Clear all data in layer"""
        pass

    def count(self) -> int:
        """Count entities in layer"""
        return 0

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from layer (for precompute)"""
        return []

    def update_embeddings(
        self,
        entity_ids: List[str],
        embeddings: List[List[float]],
        model_id: str
    ) -> int:
        """Update embeddings for entities"""
        return 0

    def map_to_record(self, raw_data: Dict[str, Any]) -> DatabaseRecord:
        """Map raw data to DatabaseRecord using field_mapping"""
        mapped = {}
        for standard_field, db_field in self.field_mapping.items():
            if db_field in raw_data:
                mapped[standard_field] = raw_data[db_field]

        # Handle embedding fields directly (not in field_mapping)
        if 'embedding' in raw_data:
            mapped['embedding'] = raw_data['embedding']
        if 'embedding_model_id' in raw_data:
            mapped['embedding_model_id'] = raw_data['embedding_model_id']

        mapped['source'] = self.config.type
        return DatabaseRecord(**mapped)


class DictLayer(DatabaseLayer):
    """Simple dict-based storage for small entity sets (<5000)"""
    
    def _setup(self):
        self._storage: Dict[str, DatabaseRecord] = {}
        self._label_index: Dict[str, str] = {}
        self._alias_index: Dict[str, Set[str]] = {}
    
    def search(self, query: str) -> List[DatabaseRecord]:
        """Fast O(1) exact search using indexes"""
        query_key = self.normalize_query(query)
        results = []
        seen = set()
        
        # Label lookup
        if query_key in self._label_index:
            eid = self._label_index[query_key]
            results.append(self._storage[eid])
            seen.add(eid)
        
        # Alias lookup
        if query_key in self._alias_index:
            for eid in self._alias_index[query_key]:
                if eid not in seen:
                    results.append(self._storage[eid])
                    seen.add(eid)
        
        return results
    
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Simple fuzzy search for small datasets (O(n) is fine for <5000 entities)"""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            print("[WARN DictLayer] rapidfuzz not installed, fuzzy search disabled")
            return []
        
        query_key = self.normalize_query(query)
        results = []
        
        # Check prefix requirement
        if self.fuzzy_config.prefix_length > 0:
            prefix = query_key[:self.fuzzy_config.prefix_length]
        
        for entity in self._storage.values():
            # Check label
            label_key = entity.label.lower()
            
            if self.fuzzy_config.prefix_length > 0:
                if not label_key.startswith(prefix):
                    continue
            
            similarity = fuzz.ratio(query_key, label_key) / 100.0
            if similarity >= self.fuzzy_config.min_similarity:
                results.append((entity, similarity))
                continue
            
            # Check aliases
            for alias in entity.aliases:
                alias_key = alias.lower()
                if self.fuzzy_config.prefix_length > 0:
                    if not alias_key.startswith(prefix):
                        continue
                
                sim = fuzz.ratio(query_key, alias_key) / 100.0
                if sim >= self.fuzzy_config.min_similarity:
                    results.append((entity, sim))
                    break
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]
    
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        """Write is same as load_bulk for dict layer"""
        self.load_bulk(records, overwrite=True)
    
    def load_bulk(self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000) -> int:
        """Bulk load entities with indexing"""
        count = 0
        for entity in entities:
            entity_id = entity.entity_id
            
            if not overwrite and entity_id in self._storage:
                continue
            
            # Store entity
            self._storage[entity_id] = entity
            
            # Index by label
            label_key = entity.label.lower()
            self._label_index[label_key] = entity_id
            
            # Index by aliases
            for alias in entity.aliases:
                alias_key = alias.lower()
                if alias_key not in self._alias_index:
                    self._alias_index[alias_key] = set()
                self._alias_index[alias_key].add(entity_id)
            
            count += 1
        return count
    
    def clear(self):
        """Clear all data"""
        self._storage.clear()
        self._label_index.clear()
        self._alias_index.clear()
    
    def count(self) -> int:
        """Count entities"""
        return len(self._storage)

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from storage"""
        return list(self._storage.values())

    def update_embeddings(
        self,
        entity_ids: List[str],
        embeddings: List[List[float]],
        model_id: str
    ) -> int:
        """Update embeddings for entities"""
        count = 0
        for eid, emb in zip(entity_ids, embeddings):
            if eid in self._storage:
                self._storage[eid].embedding = emb
                self._storage[eid].embedding_model_id = model_id
                count += 1
        return count

    def is_available(self) -> bool:
        """Dict layer is always available"""
        return True


class RedisLayer(DatabaseLayer):
    """Redis cache layer"""
    
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
    
    def load_bulk(self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000) -> int:
        """Bulk load to Redis"""
        count = 0
        pipe = self.client.pipeline()
        
        for entity in entities:
            # Prepare data
            entity_data = entity.dict()
            data_json = json.dumps(entity_data)
            
            # Store by label
            label_key = f"entity:{entity.label.lower()}"
            if overwrite or not self.client.exists(label_key):
                pipe.setex(label_key, self.ttl, data_json)
                count += 1
            
            # Store by aliases
            for alias in entity.aliases:
                alias_key = f"entity:{alias.lower()}"
                if overwrite or not self.client.exists(alias_key):
                    pipe.setex(alias_key, self.ttl, data_json)
            
            # Execute in batches
            if len(pipe) >= batch_size:
                pipe.execute()
                pipe = self.client.pipeline()
        
        # Execute remaining
        if len(pipe) > 0:
            pipe.execute()
        
        return count
    
    def clear(self):
        """Clear all entity keys"""
        for key in self.client.scan_iter(match="entity:*"):
            self.client.delete(key)
    
    def count(self) -> int:
        """Count entity keys"""
        return sum(1 for _ in self.client.scan_iter(match="entity:*"))

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from Redis (scans all entity:* keys)"""
        entities = []
        seen_ids = set()

        for key in self.client.scan_iter(match="entity:*"):
            try:
                data = self.client.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    record_data = json.loads(data)

                    if isinstance(record_data, dict):
                        if record_data.get('entity_id') not in seen_ids:
                            record_data['source'] = 'redis'
                            entities.append(DatabaseRecord(**record_data))
                            seen_ids.add(record_data.get('entity_id'))
                    elif isinstance(record_data, list):
                        for r in record_data:
                            if r.get('entity_id') not in seen_ids:
                                r['source'] = 'redis'
                                entities.append(DatabaseRecord(**r))
                                seen_ids.add(r.get('entity_id'))
            except Exception as e:
                continue

        return entities

    def update_embeddings(
        self,
        entity_ids: List[str],
        embeddings: List[List[float]],
        model_id: str
    ) -> int:
        """Update embeddings in Redis entities"""
        count = 0
        id_to_embedding = dict(zip(entity_ids, embeddings))

        for key in self.client.scan_iter(match="entity:*"):
            try:
                data = self.client.get(key)
                if not data:
                    continue

                if isinstance(data, bytes):
                    data = data.decode('utf-8')

                record_data = json.loads(data)
                updated = False

                if isinstance(record_data, dict):
                    if record_data.get('entity_id') in id_to_embedding:
                        record_data['embedding'] = id_to_embedding[record_data['entity_id']]
                        record_data['embedding_model_id'] = model_id
                        updated = True
                elif isinstance(record_data, list):
                    for r in record_data:
                        if r.get('entity_id') in id_to_embedding:
                            r['embedding'] = id_to_embedding[r['entity_id']]
                            r['embedding_model_id'] = model_id
                            updated = True

                if updated:
                    self.client.setex(key, self.ttl, json.dumps(record_data))
                    count += 1

            except Exception as e:
                continue

        return count

    def is_available(self) -> bool:
        try:
            self.client.ping()
            return True
        except:
            return False


class ElasticsearchLayer(DatabaseLayer):
    """Elasticsearch full-text search layer"""
    
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
            actions = []
            for record in records:
                doc = self._map_from_record(record)
                actions.append({
                    "_index": self.index_name,
                    "_id": record.entity_id,
                    "_source": doc
                })
            
            if actions:
                es_bulk(self.client, actions)
                self.client.indices.refresh(index=self.index_name)
        except Exception as e:
            print(f"[ERROR ES] Write error: {e}")
    
    def load_bulk(self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000) -> int:
        """Bulk load to Elasticsearch"""
        actions = []
        for entity in entities:
            doc = self._map_from_record(entity)
            
            action = {
                '_index': self.index_name,
                '_id': entity.entity_id,
                '_source': doc
            }
            
            if overwrite:
                action['_op_type'] = 'index'
            else:
                action['_op_type'] = 'create'
            
            actions.append(action)
        
        success, failed = es_bulk(
            self.client,
            actions,
            raise_on_error=False,
            chunk_size=batch_size
        )
        
        self.client.indices.refresh(index=self.index_name)
        return success
    
    def _map_from_record(self, record: DatabaseRecord) -> dict:
        """Map DatabaseRecord -> ES document using field_mapping"""
        reverse_mapping = {v: k for k, v in self.field_mapping.items()}
        
        doc = {}
        for standard_field, value in record.dict().items():
            if standard_field == 'source':
                continue
            
            es_field = reverse_mapping.get(standard_field, standard_field)
            doc[es_field] = value
        
        return doc
    
    def clear(self):
        """Delete all documents in index"""
        try:
            self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}}
            )
            self.client.indices.refresh(index=self.index_name)
        except Exception as e:
            print(f"[ERROR ES] Clear error: {e}")
    
    def count(self) -> int:
        """Count documents in index"""
        try:
            result = self.client.count(index=self.index_name)
            return result['count']
        except:
            return 0

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from Elasticsearch using scroll"""
        entities = []

        try:
            # Use scroll API for large datasets
            response = self.client.search(
                index=self.index_name,
                body={"query": {"match_all": {}}, "size": 1000},
                scroll='2m'
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while hits:
                entities.extend(self._process_hits(hits))

                response = self.client.scroll(scroll_id=scroll_id, scroll='2m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            # Clear scroll
            self.client.clear_scroll(scroll_id=scroll_id)

        except Exception as e:
            print(f"[ERROR ES] get_all_entities error: {e}")

        return entities

    def update_embeddings(
        self,
        entity_ids: List[str],
        embeddings: List[List[float]],
        model_id: str
    ) -> int:
        """Update embeddings in Elasticsearch"""
        try:
            actions = []
            for eid, emb in zip(entity_ids, embeddings):
                actions.append({
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": eid,
                    "doc": {
                        "embedding": emb,
                        "embedding_model_id": model_id
                    }
                })

            success, failed = es_bulk(
                self.client,
                actions,
                raise_on_error=False,
                chunk_size=500
            )

            self.client.indices.refresh(index=self.index_name)
            return success

        except Exception as e:
            print(f"[ERROR ES] update_embeddings error: {e}")
            return 0

    def is_available(self) -> bool:
        try:
            return self.client.ping()
        except:
            return False


class PostgresLayer(DatabaseLayer):
    """PostgreSQL database layer"""
    
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
    
    def load_bulk(self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000) -> int:
        """Bulk load to Postgres"""
        cursor = self.conn.cursor()
        
        try:
            # Prepare entity data
            entity_values = [
                (e.entity_id, e.label, e.description, e.entity_type, e.popularity)
                for e in entities
            ]
            
            # Insert entities
            if overwrite:
                entity_query = """
                    INSERT INTO entities (entity_id, label, description, entity_type, popularity)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO UPDATE SET
                        label = EXCLUDED.label,
                        description = EXCLUDED.description,
                        entity_type = EXCLUDED.entity_type,
                        popularity = EXCLUDED.popularity
                """
            else:
                entity_query = """
                    INSERT INTO entities (entity_id, label, description, entity_type, popularity)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO NOTHING
                """
            
            execute_batch(cursor, entity_query, entity_values, page_size=batch_size)
            
            # Prepare alias data
            alias_values = []
            for entity in entities:
                for alias in entity.aliases:
                    alias_values.append((entity.entity_id, alias))
            
            # Delete old aliases if overwrite
            if overwrite and alias_values:
                entity_ids = [e.entity_id for e in entities]
                cursor.execute(
                    "DELETE FROM aliases WHERE entity_id = ANY(%s)",
                    (entity_ids,)
                )
            
            # Insert aliases
            if alias_values:
                execute_batch(
                    cursor,
                    "INSERT INTO aliases (entity_id, alias) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    alias_values,
                    page_size=batch_size
                )
            
            self.conn.commit()
            return len(entities)
        
        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR Postgres] Load bulk failed: {e}")
            raise
        finally:
            cursor.close()
    
    def clear(self):
        """Clear all data"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("TRUNCATE entities, aliases CASCADE")
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR Postgres] Clear error: {e}")
        finally:
            cursor.close()
    
    def count(self) -> int:
        """Count entities"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM entities")
            return cursor.fetchone()[0]
        except:
            return 0
        finally:
            cursor.close()

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from PostgreSQL"""
        entities = []

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            sql = """
                SELECT
                    e.entity_id,
                    e.label,
                    e.description,
                    e.entity_type,
                    e.popularity,
                    e.embedding,
                    e.embedding_model_id,
                    COALESCE(array_agg(a.alias) FILTER (WHERE a.alias IS NOT NULL), ARRAY[]::text[]) as aliases
                FROM entities e
                LEFT JOIN aliases a ON e.entity_id = a.entity_id
                GROUP BY e.entity_id, e.label, e.description, e.entity_type, e.popularity, e.embedding, e.embedding_model_id
            """
            cursor.execute(sql)

            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['source'] = 'postgres'

                # Deserialize embedding from bytes if needed
                if row_dict.get('embedding'):
                    import pickle
                    if isinstance(row_dict['embedding'], (bytes, memoryview)):
                        row_dict['embedding'] = pickle.loads(bytes(row_dict['embedding']))

                record = self.map_to_record(row_dict)
                entities.append(record)

            cursor.close()

        except Exception as e:
            print(f"[ERROR Postgres] get_all_entities error: {e}")

        return entities

    def update_embeddings(
        self,
        entity_ids: List[str],
        embeddings: List[List[float]],
        model_id: str
    ) -> int:
        """Update embeddings in PostgreSQL"""
        cursor = self.conn.cursor()

        try:
            import pickle

            # Prepare batch data
            batch_data = []
            for eid, emb in zip(entity_ids, embeddings):
                emb_bytes = pickle.dumps(emb)
                batch_data.append((emb_bytes, model_id, eid))

            # Batch update
            execute_batch(
                cursor,
                "UPDATE entities SET embedding = %s, embedding_model_id = %s WHERE entity_id = %s",
                batch_data,
                page_size=500
            )

            self.conn.commit()
            return len(batch_data)

        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR Postgres] update_embeddings error: {e}")
            return 0
        finally:
            cursor.close()

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
            
            if layer_config.type == "dict":
                layer = DictLayer(layer_config)
            elif layer_config.type == "redis":
                layer = RedisLayer(layer_config)
            elif layer_config.type == "elasticsearch":
                layer = ElasticsearchLayer(layer_config)
            elif layer_config.type == "postgres":
                layer = PostgresLayer(layer_config)
            else:
                raise ValueError(f"Unknown layer type: {layer_config.type}")
            
            self.layers.append(layer)
        
        self.layers.sort(key=lambda x: x.priority, reverse=True)  # Higher priority checked first
    
    def get_available_methods(self) -> List[str]:
        return [
            "search",
            "filter_by_popularity",
            "deduplicate_candidates",
            "limit_candidates",
            "sort_by_popularity"
        ]
    
    def search(self, mention: str) -> List[DatabaseRecord]:
        """Search through layers with fallback"""
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
        """Write results to upper layers (higher priority = checked earlier)"""
        for layer in self.layers:
            # Skip source layer and all layers with lower priority
            if layer.priority <= source_layer.priority:
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
    
    def load_entities(
        self,
        filepath: str,
        target_layers: List[str] = None,
        batch_size: int = 1000,
        overwrite: bool = False
    ) -> Dict[str, int]:
        """
        Load entities from JSONL file
        
        JSONL format (DatabaseRecord):
            {"entity_id": "Q1", "label": "Python", "aliases": [...], "popularity": 1000000, ...}
        
        Args:
            filepath: path to .jsonl file
            target_layers: ['dict', 'redis', 'elasticsearch', 'postgres'] or None (all writable)
            batch_size: batch size for bulk operations
            overwrite: overwrite existing entities
        
        Returns:
            {'redis': 1500, 'elasticsearch': 1500}
        """
        print(f"\nLoading entities from {filepath}...")
        
        # Parse JSONL
        entities = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    entity = DatabaseRecord(**data)
                    entities.append(entity)
                except Exception as e:
                    print(f"[WARN] Line {line_num} parse error: {e}")
                    continue
        
        print(f"✓ Loaded {len(entities)} entities from file")
        
        # Determine target layers
        if target_layers is None:
            target_layers = [l.config.type for l in self.layers if l.write]
        
        # Load to each layer
        results = {}
        for layer in self.layers:
            if layer.config.type not in target_layers:
                continue
            
            if not layer.is_available():
                print(f"[WARN] {layer.config.type} unavailable, skipping")
                continue
            
            print(f"\nLoading to {layer.config.type}...")
            count = layer.load_bulk(entities, overwrite=overwrite, batch_size=batch_size)
            results[layer.config.type] = count
            print(f"✓ Loaded {count} entities")
        
        return results
    
    def clear_layers(self, layer_names: List[str] = None):
        """Clear all entities in specified layers"""
        for layer in self.layers:
            if layer_names and layer.config.type not in layer_names:
                continue
            
            print(f"Clearing {layer.config.type}...")
            layer.clear()
            print(f"✓ Cleared")
    
    def count_entities(self) -> Dict[str, int]:
        """Count entities in each layer"""
        counts = {}
        for layer in self.layers:
            counts[layer.config.type] = layer.count()
        return counts

    def precompute_embeddings(
        self,
        encoder_fn,
        template: str,
        model_id: str,
        target_layers: List[str] = None,
        batch_size: int = 32
    ) -> Dict[str, int]:
        """
        Precompute embeddings for all entities in specified layers.

        Args:
            encoder_fn: Callable that takes List[str] and returns embeddings tensor
            template: Template string for formatting labels (e.g., "{label}: {description}")
            model_id: Model identifier to store with embeddings
            target_layers: Layer types to update (None = all)
            batch_size: Batch size for encoding

        Returns:
            Dict with count of updated entities per layer
        """
        from tqdm import tqdm

        results = {}

        for layer in self.layers:
            if target_layers and layer.config.type not in target_layers:
                continue

            if not layer.is_available():
                print(f"[WARN] {layer.config.type} unavailable, skipping")
                continue

            print(f"\nPrecomputing embeddings for {layer.config.type}...")

            # Get all entities
            entities = layer.get_all_entities()
            if not entities:
                print(f"  No entities found in {layer.config.type}")
                continue

            print(f"  Found {len(entities)} entities")

            # Format labels using template
            labels = []
            entity_ids = []
            for entity in entities:
                try:
                    formatted = template.format(**entity.dict())
                    labels.append(formatted)
                    entity_ids.append(entity.entity_id)
                except KeyError as e:
                    print(f"  [WARN] Template error for {entity.entity_id}: {e}")
                    continue

            # Encode in batches
            all_embeddings = []
            for i in tqdm(range(0, len(labels), batch_size), desc="Encoding"):
                batch_labels = labels[i:i + batch_size]
                batch_embeddings = encoder_fn(batch_labels)

                # Convert to list if tensor
                if hasattr(batch_embeddings, 'tolist'):
                    batch_embeddings = batch_embeddings.tolist()
                elif hasattr(batch_embeddings, 'cpu'):
                    batch_embeddings = batch_embeddings.cpu().numpy().tolist()

                all_embeddings.extend(batch_embeddings)

            # Update layer
            updated = layer.update_embeddings(entity_ids, all_embeddings, model_id)
            results[layer.config.type] = updated
            print(f"  Updated {updated} entities with embeddings")

        return results

    def get_layer(self, layer_type: str) -> DatabaseLayer:
        """Get layer by type"""
        for layer in self.layers:
            if layer.config.type == layer_type:
                return layer
        return None