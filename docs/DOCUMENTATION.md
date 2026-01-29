# GLiNKER Technical Documentation

**Version:** 0.1.0
**Framework:** GLiNKER (GLiNER-based Entity Linking & Knowledge Extraction Runtime)

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Design Patterns](#design-patterns)
4. [Layer System (L0-L3)](#layer-system)
5. [Configuration System](#configuration-system)
6. [Database Layers](#database-layers)
7. [Embedding System](#embedding-system)
8. [Field Resolution](#field-resolution)
9. [API Reference](#api-reference)
10. [Complete Examples](#complete-examples)

---

## Overview

GLiNKER is a four-layer entity linking pipeline framework that combines named entity recognition (NER), database search, and neural entity disambiguation into a unified, declarative system.

### Key Features

- **DAG-based Execution**: Define pipelines declaratively in YAML with automatic dependency resolution
- **Component-Processor Pattern**: Separates core logic (components) from orchestration (processors)
- **Multi-Layer Database**: Hierarchical cache system (Dict → Redis → Elasticsearch → PostgreSQL)
- **Precomputed Embeddings**: BiEncoder support for 100x faster entity disambiguation
- **Field Resolution**: JSONPath-like syntax for flexible data extraction and transformation
- **Factory Registry**: Auto-registration of processors for clean plugin architecture

### Pipeline Flow

```
L1 (NER)           L2 (Database)         L3 (Linking)         L0 (Aggregation)
--------           -------------         ------------         ----------------
Texts              Mentions              Texts                L1 + L2 + L3
  ↓                   ↓                  Candidates              ↓
Extract            Search                  ↓                 Aggregate
Entities         → Candidates →         Disambiguate    →    Statistics
  ↓                   ↓                     ↓                    ↓
[Entities]        [Candidates]         [Linked Entities]    [L0 Entities]
```

---

## Core Architecture

### 1. DAG Pipeline System

The system uses **Directed Acyclic Graphs (DAGs)** for pipeline execution, enabling parallel processing and complex dependencies.

#### Key Components

**`DAGPipeline`** (src/glinker/core/dag.py)
```python
class DAGPipeline(BaseModel):
    name: str                      # Pipeline identifier
    nodes: List[PipeNode]          # Processing stages
    description: Optional[str]     # Documentation
```

**`PipeNode`** (src/glinker/core/dag.py)
```python
class PipeNode(BaseModel):
    id: str                        # Unique node identifier (e.g., "l1", "l2")
    processor: str                 # Processor name from registry
    inputs: Dict[str, InputConfig] # Input parameter mappings
    output: OutputConfig           # Output specification
    requires: List[str]            # Explicit dependencies
    config: Dict[str, Any]         # Processor configuration
    schema: Optional[Dict]         # Field mappings/transformations
    condition: Optional[str]       # Conditional execution
```

**`DAGExecutor`** (src/glinker/core/dag.py)
```python
class DAGExecutor:
    """
    Executes DAG pipeline with:
    - Topological sorting for correct execution order
    - Processor initialization and caching (reused across executions)
    - Automatic dependency resolution from input sources
    - Persistent database connections
    """

    def __init__(self, pipeline: DAGPipeline, verbose: bool = False)
    def execute(self, pipeline_input: Any) -> PipeContext
    def load_entities(self, filepath: str, ...) -> Dict[str, Dict[str, int]]
    def precompute_embeddings(...) -> Dict[str, int]
```

**`PipeContext`** (src/glinker/core/dag.py)
```python
class PipeContext:
    """
    Stores intermediate results and provides unified data access:

    - By key: context.get("l1_result")
    - By index: context.get("outputs[-1]")  # last output
    - Pipeline input: context.get("$input")
    """

    def set(self, key: str, value: Any, metadata: Optional[Dict])
    def get(self, source: str) -> Any
    def to_json(self, filepath: str = None) -> str
    @classmethod
    def from_json(cls, filepath: str = None) -> PipeContext
```

**`FieldResolver`** (src/glinker/core/dag.py)
```python
class FieldResolver:
    """
    Extracts data using JSONPath-like syntax:

    - 'entities' → field access
    - 'entities[*]' → iterate list
    - 'entities[*][*].text' → nested iteration
    - 'entities[0]' → index access
    - 'entities[1:]' → slicing
    """

    @staticmethod
    def resolve(context: PipeContext, config: InputConfig) -> Any
```

#### Dependency Resolution

Dependencies are established **automatically** from `InputConfig` source references:

```yaml
nodes:
  - id: "l2"
    inputs:
      mentions: {source: "l1_result", ...}  # Creates l1 → l2 dependency
```

Execution order is determined by **topological sort** with level grouping for parallelization.

#### Context Serialization

Contexts can be saved for debugging:

```python
context = executor.execute(input_data)
context.to_json("debug_context.json")

# Later...
restored = PipeContext.from_json(filepath="debug_context.json")
```

---

### 2. Component-Processor Architecture

All processing follows a **separation of concerns** pattern:

- **BaseComponent**: Implements discrete, chainable methods
- **BaseProcessor**: Orchestrates component methods via configurable pipelines
- **Factory Pattern**: Auto-registration via decorators

**BaseComponent** (src/glinker/core/base.py)
```python
class BaseComponent(ABC, Generic[ConfigT]):
    """
    Implements core logic as discrete methods.
    Each method is pure and chainable.
    """

    def __init__(self, config: ConfigT)
    def _setup(self): pass  # Override for initialization

    @abstractmethod
    def get_available_methods(self) -> list[str]:
        """Return list of methods available for pipelines"""
        pass
```

**BaseProcessor** (src/glinker/core/base.py)
```python
class BaseProcessor(ABC, Generic[ConfigT, InputT, OutputT]):
    """
    Orchestrates component methods via pipeline.
    Pipeline: [(method_name, kwargs), ...]
    """

    def __init__(self, config, component, pipeline=None)

    @abstractmethod
    def _default_pipeline(self) -> list[tuple[str, dict]]:
        """Define default processing pipeline"""
        pass

    def _execute_pipeline(self, data: Any, pipeline: list) -> Any:
        """Execute pipeline on data"""
        for method_name, kwargs in pipeline:
            method = getattr(self.component, method_name)
            data = method(data, **kwargs)
        return data

    @abstractmethod
    def __call__(self, input_data: InputT) -> OutputT:
        """Main entry point"""
        pass
```

#### Example: L1 Component Pipeline

```python
class L1SpacyComponent(BaseComponent[L1Config]):
    def get_available_methods(self) -> list[str]:
        return [
            "extract_entities",    # NER extraction
            "filter_by_length",    # Remove short mentions
            "deduplicate",         # Remove duplicates
            "sort_by_position",    # Sort by position
            "add_noun_chunks"      # Add noun phrases
        ]

class L1SpacyProcessor(BaseProcessor[L1Config, L1Input, L1Output]):
    def _default_pipeline(self):
        return [
            ("extract_entities", {}),
            ("deduplicate", {}),
            ("sort_by_position", {})
        ]
```

---

### 3. Registry System

Processors are auto-registered via decorators for clean dependency injection.

**ProcessorRegistry** (src/glinker/core/registry.py)
```python
class ProcessorRegistry:
    def register(self, name: str):
        """Decorator for processor factory registration"""
        def decorator(factory: Callable):
            self._registry[name] = factory
            return factory
        return decorator

    def get(self, name: str) -> Callable:
        """Get processor factory by name"""
        if name not in self._registry:
            raise KeyError(f"Processor '{name}' not found")
        return self._registry[name]

processor_registry = ProcessorRegistry()
```

**Usage:**

```python
@processor_registry.register("l1_spacy")
def create_l1_spacy_processor(config_dict: dict, pipeline: list = None):
    config = L1Config(**config_dict)
    component = L1SpacyComponent(config)
    return L1SpacyProcessor(config, component, pipeline)
```

Processors must be imported to trigger registration:

```python
# src/glinker/l1/__init__.py
from .processor import create_l1_spacy_processor  # Triggers @register
```

---

### 4. Factory Pattern

**ProcessorFactory** (src/glinker/core/factory.py)
```python
class ProcessorFactory:
    @staticmethod
    def create_pipeline(config_path: str, verbose: bool = False) -> DAGExecutor:
        """Create DAG pipeline from YAML config"""
        config = yaml.safe_load(open(config_path))
        pipeline = DAGPipeline(**config)
        return DAGExecutor(pipeline, verbose=verbose)

    @staticmethod
    def create_from_dict(config_dict: dict, verbose: bool = False) -> DAGExecutor:
        """Create pipeline from dict (programmatic use)"""
        pipeline = DAGPipeline(**config_dict)
        return DAGExecutor(pipeline, verbose=verbose)
```

---

## Layer System

### L1: Named Entity Recognition

Extracts entity mentions from text using either spaCy or GLiNER models.

#### Models

**L1Config** (src/glinker/l1/models.py)
```python
class L1Config(BaseConfig):
    model: str = "en_core_sci_sm"       # spaCy model
    device: str = "cpu"
    batch_size: int = 16
    max_right_context: int = 50         # Context window size
    max_left_context: int = 50
    min_entity_length: int = 2          # Filter short mentions
    include_noun_chunks: bool = False   # Add noun phrases
```

**L1GlinerConfig** (src/glinker/l1/models.py)
```python
class L1GlinerConfig(L1Config):
    model: str                          # GLiNER model (required)
    labels: List[str]                   # Fixed entity types
    token: Optional[str] = None         # HuggingFace token
    threshold: float = 0.3              # Confidence threshold
    flat_ner: bool = True               # No nested entities
    multi_label: bool = False           # Single label per entity
    use_precomputed_embeddings: bool = False
    max_length: Optional[int] = None    # Max sequence length
```

**L1Entity** (src/glinker/l1/models.py)
```python
class L1Entity(BaseOutput):
    text: str              # Extracted mention text
    start: int             # Start position in text
    end: int               # End position in text
    left_context: str      # Left context
    right_context: str     # Right context
```

**L1Output** (src/glinker/l1/models.py)
```python
class L1Output(BaseOutput):
    entities: list[list[L1Entity]]  # [[text1_entities], [text2_entities], ...]
```

#### Components

**L1SpacyComponent** (src/glinker/l1/component.py)

Available methods:
- `extract_entities(text: str) -> list[L1Entity]` - Extract NER entities
- `filter_by_length(entities, min_length) -> list[L1Entity]` - Filter by length
- `deduplicate(entities) -> list[L1Entity]` - Remove duplicates
- `sort_by_position(entities) -> list[L1Entity]` - Sort by position
- `add_noun_chunks(text, entities) -> list[L1Entity]` - Add noun phrases

**L1GlinerComponent** (src/glinker/l1/component.py)

Additional methods:
- `encode_labels(labels: List[str]) -> torch.Tensor` - Encode labels (BiEncoder only)
- `supports_precomputed_embeddings: bool` - Check BiEncoder support

#### Processors

**L1SpacyProcessor** (src/glinker/l1/processor.py)
```python
class L1SpacyProcessor(BaseProcessor):
    def __call__(self, texts: List[str] = None) -> L1Output:
        """Batch process using spaCy's efficient pipe"""
        # Uses nlp.pipe() for batch processing
        # Applies pipeline to each text
        return L1Output(entities=results)
```

**L1GlinerProcessor** (src/glinker/l1/processor.py)
```python
class L1GlinerProcessor(BaseProcessor):
    def __call__(self, texts: List[str] = None) -> L1Output:
        """Batch process using GLiNER"""
        # Uses precomputed label embeddings if available
        # Falls back to regular prediction
        return L1Output(entities=results)
```

#### Configuration Example

```yaml
- id: "l1"
  processor: "l1_spacy"  # or "l1_gliner"
  inputs:
    texts: {source: "$input", fields: "texts"}
  output: {key: "l1_result"}
  config:
    model: "en_core_sci_sm"
    batch_size: 32
    min_entity_length: 3
```

---

### L2: Multi-Layer Database Search

Searches for candidate entities across multiple database layers with hierarchical caching.

#### Models

**L2Config** (src/glinker/l2/models.py)
```python
class L2Config(BaseConfig):
    layers: List[LayerConfig]           # Database layers
    max_candidates: int = 30            # Max candidates per mention
    min_popularity: int = 0             # Minimum popularity threshold
    embeddings: Optional[EmbeddingConfig] = None  # Embedding support
```

**LayerConfig** (src/glinker/l2/models.py)
```python
class LayerConfig(BaseConfig):
    type: str                           # "dict", "redis", "elasticsearch", "postgres"
    priority: int                       # Search priority (higher = first)
    config: Dict[str, Any]              # Layer-specific config
    search_mode: List[str]              # ["exact"], ["fuzzy"], or ["exact", "fuzzy"]
    write: bool = True                  # Enable cache writes
    cache_policy: str = "always"        # "always", "miss", "hit"
    ttl: int = 3600                     # Cache TTL (seconds, 0 = no expiry)
    field_mapping: Dict[str, str]       # Standard field → DB field mapping
    fuzzy: Optional[FuzzyConfig]        # Fuzzy search config
```

**FuzzyConfig** (src/glinker/l2/models.py)
```python
class FuzzyConfig(BaseConfig):
    max_distance: int = 2               # Max Levenshtein distance
    min_similarity: float = 0.3         # Min similarity threshold
    n_gram_size: int = 3                # N-gram size
    prefix_length: int = 1              # Prefix to preserve
```

**EmbeddingConfig** (src/glinker/l2/models.py)
```python
class EmbeddingConfig(BaseModel):
    enabled: bool = False               # Enable embedding support
    model_name: Optional[str] = None    # Model for encoding
    dim: int = 768                      # Embedding dimension
    precompute_on_load: bool = False    # Compute during load_bulk
    batch_size: int = 32                # Encoding batch size
```

**DatabaseRecord** (src/glinker/l2/models.py)
```python
class DatabaseRecord(BaseModel):
    """Unified format for all database layers"""
    entity_id: str                      # Unique identifier
    label: str                          # Primary name
    aliases: List[str]                  # Alternative names
    description: str = ""               # Description
    entity_type: str = ""               # Category/type
    popularity: int = 0                 # Popularity score
    metadata: Dict[str, Any]            # Additional metadata
    source: str = ""                    # Source layer

    # Embedding fields (for BiEncoder)
    embedding: Optional[List[float]] = None
    embedding_model_id: Optional[str] = None
```

**L2Output** (src/glinker/l2/models.py)
```python
class L2Output(BaseOutput):
    candidates: List[List[DatabaseRecord]]  # [[text1_candidates], [text2_candidates], ...]
```

#### Database Layers

**DatabaseLayer** (src/glinker/l2/component.py)

Abstract base class for all layers:

```python
class DatabaseLayer(ABC):
    def __init__(self, config: LayerConfig)

    @abstractmethod
    def search(self, query: str) -> List[DatabaseRecord]:
        """Exact search"""
        pass

    @abstractmethod
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Fuzzy search"""
        pass

    @abstractmethod
    def load_bulk(self, entities: List[DatabaseRecord], ...) -> int:
        """Bulk load entities"""
        pass

    @abstractmethod
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        """Write to cache"""
        pass

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities (for precompute)"""
        return []

    def update_embeddings(self, entity_ids, embeddings, model_id) -> int:
        """Update precomputed embeddings"""
        return 0
```

**DictLayer** (src/glinker/l2/component.py)

In-memory dictionary for small datasets (<5000 entities):

```python
class DictLayer(DatabaseLayer):
    """
    Fast O(1) lookups using indexes:
    - _storage: {entity_id → DatabaseRecord}
    - _label_index: {label_lower → entity_id}
    - _alias_index: {alias_lower → Set[entity_id]}

    Fuzzy search: O(n) using rapidfuzz (acceptable for small datasets)
    """

    def search(self, query: str) -> List[DatabaseRecord]:
        """Exact search via indexes"""
        # Check label index
        # Check alias index
        return results

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Fuzzy search using rapidfuzz"""
        # Linear scan with similarity check
        # Sort by similarity
        return results
```

**RedisLayer** (src/glinker/l2/component.py)

Fast cache with TTL support:

```python
class RedisLayer(DatabaseLayer):
    """
    Redis cache layer (exact match only)

    Keys: entity:{query_lower}
    Values: JSON-encoded list of DatabaseRecord
    """

    def search(self, query: str) -> List[DatabaseRecord]:
        """Get from Redis by key"""
        key = f"entity:{query.lower()}"
        data = self.client.get(key)
        return json.loads(data) if data else []

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Not supported"""
        return []

    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        """Write with TTL"""
        self.client.setex(f"entity:{key.lower()}", ttl, json.dumps(records))
```

**ElasticsearchLayer** (src/glinker/l2/component.py)

Full-text search with fuzzy matching:

```python
class ElasticsearchLayer(DatabaseLayer):
    """
    Elasticsearch full-text search

    Supports:
    - Multi-field search (label, aliases, description)
    - Fuzzy matching with configurable distance
    - Field boosting (label^2, aliases^1.5)
    """

    def search(self, query: str) -> List[DatabaseRecord]:
        """Multi-field best_fields query"""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["label^2", "aliases^1.5", "description"],
                    "type": "best_fields"
                }
            }
        }
        return self._process_hits(response['hits']['hits'])

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Fuzzy multi-field query"""
        # Add fuzziness, prefix_length, max_expansions
        return results
```

**PostgresLayer** (src/glinker/l2/component.py)

Primary storage with pg_trgm for fuzzy search:

```python
class PostgresLayer(DatabaseLayer):
    """
    PostgreSQL primary storage

    Tables:
    - entities: (entity_id, label, description, type, popularity, embedding, ...)
    - aliases: (entity_id, alias)

    Uses pg_trgm extension for trigram similarity matching
    """

    def search(self, query: str) -> List[DatabaseRecord]:
        """Exact search with LIKE"""
        sql = """
            SELECT e.*, ARRAY_AGG(a.alias) as aliases
            FROM entities e
            LEFT JOIN aliases a ON e.entity_id = a.entity_id
            WHERE LOWER(e.label) LIKE %s OR ...
            GROUP BY e.entity_id
        """
        return results

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Fuzzy search using pg_trgm similarity()"""
        sql = """
            SELECT ..., similarity(LOWER(e.label), %s) AS sim_score
            FROM entities e
            WHERE similarity(LOWER(e.label), %s) >= %s
            ORDER BY sim_score DESC
        """
        return results
```

#### DatabaseChainComponent

Orchestrates multi-layer search with automatic cache write-back:

```python
class DatabaseChainComponent(BaseComponent[L2Config]):
    """
    Multi-layer database chain with cache hierarchy

    Search flow:
    1. Check layers in priority order (highest first)
    2. On hit: return results, write to upper layers
    3. On miss: continue to next layer
    4. Cache write-back controlled by cache_policy
    """

    def get_available_methods(self) -> List[str]:
        return [
            "search",                   # Multi-layer search
            "filter_by_popularity",     # Filter by popularity
            "deduplicate_candidates",   # Remove duplicates
            "limit_candidates",         # Limit to max_candidates
            "sort_by_popularity"        # Sort by popularity
        ]

    def search(self, mention: str) -> List[DatabaseRecord]:
        """Search through layers with fallback and cache write-back"""
        for layer in sorted(self.layers, key=lambda x: x.priority, reverse=True):
            results = []
            for mode in layer.config.search_mode:
                if mode == "exact":
                    results.extend(layer.search(mention))
                elif mode == "fuzzy" and layer.supports_fuzzy():
                    results.extend(layer.search_fuzzy(mention))

            if results:
                self._cache_write(mention, results, source_layer=layer)
                return results

        return []

    def _cache_write(self, query, results, source_layer):
        """Write results to upper layers (priority > source_layer.priority)"""
        for layer in self.layers:
            if layer.priority <= source_layer.priority:
                continue
            if not layer.write:
                continue

            if layer.cache_policy == "always":
                layer.write_cache(query, results, layer.ttl)
            elif layer.cache_policy == "miss":
                if not layer.search(query):
                    layer.write_cache(query, results, layer.ttl)
            elif layer.cache_policy == "hit":
                if layer.search(query):
                    layer.write_cache(query, results, layer.ttl)
```

#### L2 Processor

```python
class L2Processor(BaseProcessor):
    def __call__(self, mentions: Union[List[str], List[List[Any]]]) -> L2Output:
        """
        Process mentions and return candidates

        Supports:
        - Flat list: ["mention1", "mention2", ...]
        - Nested list: [[L1Entity, ...], [L1Entity, ...]]
        """
        # Execute pipeline for each mention
        # Group by structure if provided
        return L2Output(candidates=grouped)
```

#### Cache Policies

- **`always`**: Write to cache on every search
- **`miss`**: Write only if key doesn't exist (avoid overwriting fresh data)
- **`hit`**: Write only if key exists (refresh TTL for popular queries)

#### Configuration Example

```yaml
- id: "l2"
  processor: "l2_chain"
  inputs:
    mentions: {source: "l1_result", fields: "entities"}
  output: {key: "l2_result"}
  schema:
    template: "{label}: {description}"
  config:
    max_candidates: 10
    min_popularity: 0
    layers:
      - type: "dict"
        priority: 3
        write: true
        search_mode: ["exact", "fuzzy"]
        ttl: 0
        cache_policy: "always"
        fuzzy:
          min_similarity: 0.6

      - type: "redis"
        priority: 2
        write: true
        search_mode: ["exact"]
        ttl: 3600
        cache_policy: "always"
        config:
          host: "localhost"
          port: 6379
          db: 0

      - type: "elasticsearch"
        priority: 1
        write: true
        search_mode: ["exact", "fuzzy"]
        ttl: 86400
        cache_policy: "miss"
        config:
          hosts: ["http://localhost:9200"]
          index_name: "entities"
        fuzzy:
          min_similarity: 0.3

      - type: "postgres"
        priority: 0
        write: false
        search_mode: ["exact", "fuzzy"]
        config:
          host: "localhost"
          port: 5432
          database: "entities_db"
          user: "postgres"
          password: "postgres"
        fuzzy:
          min_similarity: 0.3
```

---

### L3: Entity Disambiguation

Links mentions to candidates using GLiNER zero-shot entity linking.

#### Models

**L3Config** (src/glinker/l3/models.py)
```python
class L3Config(BaseConfig):
    model_name: str                     # GLiNER model
    token: str = None                   # HuggingFace token
    device: str = "cpu"
    threshold: float = 0.5              # Confidence threshold
    flat_ner: bool = True               # No nested entities
    multi_label: bool = False           # Single label per entity
    batch_size: int = 8                 # Batch size

    # Embedding settings
    use_precomputed_embeddings: bool = True   # Use L2 embeddings
    cache_embeddings: bool = False            # Cache computed embeddings
    max_length: Optional[int] = None          # Max sequence length
```

**L3Entity** (src/glinker/l3/models.py)
```python
class L3Entity(BaseOutput):
    text: str              # Matched text
    label: str             # Linked entity label
    start: int             # Start position
    end: int               # End position
    score: float           # Confidence score
```

**L3Output** (src/glinker/l3/models.py)
```python
class L3Output(BaseOutput):
    entities: List[List[L3Entity]]  # [[text1_entities], [text2_entities], ...]
```

#### Component

**L3Component** (src/glinker/l3/component.py)

```python
class L3Component(BaseComponent[L3Config]):
    """GLiNER-based entity linking component"""

    @property
    def supports_precomputed_embeddings(self) -> bool:
        """Check if model is BiEncoder (supports label precomputation)"""
        return (hasattr(self.model, 'encode_labels') and
                self.model.config.labels_encoder is not None)

    def get_available_methods(self) -> List[str]:
        return [
            "predict_entities",          # Standard prediction
            "predict_with_embeddings",   # Use precomputed embeddings
            "encode_labels",             # Encode labels (BiEncoder)
            "filter_by_score",           # Filter by confidence
            "sort_by_position",          # Sort by position
            "deduplicate_entities"       # Remove duplicates
        ]

    def encode_labels(self, labels: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode labels using GLiNER's BiEncoder

        Returns: (num_labels, hidden_size) tensor
        """
        if not self.supports_precomputed_embeddings:
            raise NotImplementedError("Only BiEncoder models support this")

        return self.model.encode_labels(labels, batch_size=batch_size)

    def predict_with_embeddings(
        self,
        text: str,
        labels: List[str],
        embeddings: torch.Tensor
    ) -> List[L3Entity]:
        """
        Fast prediction using precomputed label embeddings

        100x faster than regular prediction for large label sets
        """
        entities = self.model.predict_with_embeds(
            text, embeddings, labels,
            threshold=self.config.threshold,
            flat_ner=self.config.flat_ner,
            multi_label=self.config.multi_label
        )
        return [L3Entity(**e) for e in entities]

    def predict_entities(self, text: str, labels: List[str]) -> List[L3Entity]:
        """Standard prediction (computes embeddings on-the-fly)"""
        entities = self.model.predict_entities(
            text, labels,
            threshold=self.config.threshold,
            flat_ner=self.config.flat_ner,
            multi_label=self.config.multi_label
        )
        return [L3Entity(**e) for e in entities]
```

#### Processor

**L3Processor** (src/glinker/l3/processor.py)

```python
class L3Processor(BaseProcessor):
    def __call__(
        self,
        texts: List[str],
        candidates: List[List[Any]]
    ) -> L3Output:
        """
        Link entities using GLiNER

        Flow:
        1. Format candidates using schema template
        2. Check if precomputed embeddings available
        3. Use predict_with_embeddings() or predict_entities()
        4. Apply ranking if configured
        """

        for text, text_candidates in zip(texts, candidates):
            # Create labels from candidates using schema template
            labels, label_to_candidate = self._create_gliner_labels_with_mapping(
                text_candidates
            )

            # Check if we can use precomputed embeddings
            use_precomputed = (
                self.config.use_precomputed_embeddings and
                self.component.supports_precomputed_embeddings and
                self._can_use_precomputed(text_candidates, label_to_candidate)
            )

            if use_precomputed:
                # Get embeddings from candidates (stored in L2)
                embeddings = self._get_embeddings_tensor(
                    text_candidates, labels, label_to_candidate
                )
                entities = self.component.predict_with_embeddings(
                    text, labels, embeddings
                )
            else:
                # Regular prediction
                entities = self.component.predict_entities(text, labels)

                # Optionally cache computed embeddings
                if self.config.cache_embeddings:
                    self._cache_embeddings(text_candidates, labels)

            # Apply ranking if configured
            if self.schema.get('ranking'):
                entities = self._rank_entities(entities, text_candidates)

            all_entities.append(entities)

        return L3Output(entities=all_entities)

    def _can_use_precomputed(self, candidates, label_to_candidate) -> bool:
        """Check if all candidates have compatible embeddings"""
        expected_model = self.config.model_name

        for candidate in candidates:
            if not hasattr(candidate, 'embedding') or candidate.embedding is None:
                return False
            if getattr(candidate, 'embedding_model_id', None) != expected_model:
                return False

        return True

    def _rank_entities(self, entities, candidates) -> List[L3Entity]:
        """Re-rank using weighted scoring"""
        # Combine GLiNER score + popularity + other factors
        # Update entity.score with weighted average
        return sorted(entities, key=lambda x: x.score, reverse=True)
```

#### Schema Template

The `schema.template` is critical for L3 - it formats candidate labels for GLiNER:

```yaml
schema:
  template: "{label}: {description}"  # "TP53: Tumor protein p53..."
```

This template is used to:
1. Format candidates into GLiNER labels
2. Match L3 outputs back to L2 candidates
3. Ensure consistency across L2/L3/L0 layers

#### Ranking

Multi-factor ranking combines GLiNER score with candidate metadata:

```yaml
schema:
  template: "{label}"
  ranking:
    - field: "gliner_score"
      weight: 0.7
    - field: "popularity"
      weight: 0.3
```

#### Configuration Example

```yaml
- id: "l3"
  processor: "l3_batch"
  inputs:
    texts: {source: "$input", fields: "texts"}
    candidates: {source: "l2_result", fields: "candidates"}
  output: {key: "l3_result"}
  schema:
    template: "{label}: {description}"
  config:
    model_name: "knowledgator/gliner-linker-large-v1.0"
    device: "cpu"
    threshold: 0.5
    batch_size: 1
    use_precomputed_embeddings: true
    cache_embeddings: false
```

---

### L0: Final Aggregation

Combines outputs from L1, L2, L3 into unified structures with statistics.

#### Models

**L0Config** (src/glinker/l0/models.py)
```python
class L0Config(BaseConfig):
    min_confidence: float = 0.0         # Min confidence for linked entities
    include_unlinked: bool = True       # Include mentions without links
    return_all_candidates: bool = False # Return all or only top match
    strict_matching: bool = True        # Only L1-matched entities
    position_tolerance: int = 2         # Fuzzy position matching tolerance
```

**LinkedEntity** (src/glinker/l0/models.py)
```python
class LinkedEntity(BaseOutput):
    entity_id: str         # From matched candidate
    label: str             # Entity label
    confidence: float      # L3 confidence score
    start: int             # Position in text
    end: int
    matched_text: str      # Text from L3
```

**L0Entity** (src/glinker/l0/models.py)
```python
class L0Entity(BaseOutput):
    """
    Aggregated entity combining all layers:
    - L1: mention detection
    - L2: candidates
    - L3: disambiguation
    """

    # From L1
    mention_text: str
    mention_start: int
    mention_end: int
    left_context: str
    right_context: str

    # From L2
    candidates: List[DatabaseRecord]
    num_candidates: int

    # From L3
    linked_entity: Optional[LinkedEntity]
    is_linked: bool

    # Metadata
    pipeline_stage: str  # "l1_only", "l2_found", "l3_linked", "l3_only"
```

**L0Output** (src/glinker/l0/models.py)
```python
class L0Output(BaseOutput):
    entities: List[List[L0Entity]]  # [[text1_entities], [text2_entities], ...]
    stats: dict                     # Pipeline statistics
```

#### Component

**L0Component** (src/glinker/l0/component.py)

```python
class L0Component(BaseComponent[L0Config]):
    """
    Aggregation workflow:
    1. For each L1 mention → find L2 candidates
    2. For each L1 mention → check if linked in L3
    3. Create L0Entity with full information
    4. If loose mode → include L3 entities outside L1 mentions
    """

    def get_available_methods(self) -> List[str]:
        return [
            "aggregate",             # Main aggregation
            "filter_by_confidence",  # Filter by confidence
            "sort_by_confidence",    # Sort by confidence
            "calculate_stats"        # Calculate statistics
        ]

    def aggregate(
        self,
        l1_entities: List[List[L1Entity]],
        l2_candidates: List[List[DatabaseRecord]],
        l3_entities: List[List[L3Entity]],
        template: str = "{label}"
    ) -> List[List[L0Entity]]:
        """
        Main aggregation method

        Matches:
        - L1 mention → L2 candidates (by text)
        - L1 mention → L3 entity (by position)
        - L3 entity → L2 candidate (by formatted label)
        """
        all_results = []

        for text_idx in range(len(l1_entities)):
            l1_mentions = l1_entities[text_idx]
            l2_cands = l2_candidates[text_idx]
            l3_links = l3_entities[text_idx]

            text_results = self._aggregate_single_text(
                l1_mentions, l2_cands, l3_links, template
            )
            all_results.append(text_results)

        return all_results

    def _aggregate_single_text(...) -> List[L0Entity]:
        """Aggregate for single text"""
        # Build L3 index by position
        l3_by_position = {(e.start, e.end): e for e in l3_links}

        results = []
        used_l3_positions = set()

        for l1_mention in l1_mentions:
            # Get candidates for this mention
            mention_candidates = self._get_candidates_for_mention(
                l1_mention, l2_candidates
            )

            # Find linked entity
            linked_entity, l3_pos = self._find_linked_entity_with_position(
                l1_mention, l3_by_position, mention_candidates, template
            )

            if l3_pos:
                used_l3_positions.add(l3_pos)

            # Create L0Entity
            pipeline_stage = self._determine_stage(mention_candidates, linked_entity)

            l0_entity = L0Entity(
                mention_text=l1_mention.text,
                mention_start=l1_mention.start,
                mention_end=l1_mention.end,
                left_context=l1_mention.left_context,
                right_context=l1_mention.right_context,
                candidates=mention_candidates,
                num_candidates=len(mention_candidates),
                linked_entity=linked_entity,
                is_linked=linked_entity is not None,
                pipeline_stage=pipeline_stage
            )

            results.append(l0_entity)

        # Include L3-only entities if loose matching
        if not self.config.strict_matching:
            for (start, end), l3_entity in l3_by_position.items():
                if (start, end) not in used_l3_positions:
                    # L3 found entity without L1 mention
                    results.append(self._create_l3_only_entity(l3_entity, l2_candidates))

        return results

    def _find_linked_entity_with_position(...):
        """
        Find L3 entity for L1 mention

        Matching strategy:
        1. Exact position match: (l1.start, l1.end) == (l3.start, l3.end)
        2. Fuzzy position match: within position_tolerance
        3. Label match using schema template
        """
        # Try exact position
        key = (l1_mention.start, l1_mention.end)
        l3_entity = l3_by_position.get(key)

        if not l3_entity:
            # Try fuzzy position
            l3_entity = self._fuzzy_position_match(
                l1_mention.start, l1_mention.end, l3_by_position
            )

        if not l3_entity:
            return None, None

        # Match L3 label to L2 candidate using template
        matched_candidate = self._match_candidate_by_label(
            l3_entity.label, candidates, template
        )

        return LinkedEntity(
            entity_id=matched_candidate.entity_id,
            label=matched_candidate.label,
            confidence=l3_entity.score,
            start=l3_entity.start,
            end=l3_entity.end,
            matched_text=l3_entity.text
        ), key

    def _match_candidate_by_label(l3_label, candidates, template):
        """
        Match L3 label to L2 candidate

        Uses same template formatting as L3 to ensure exact match:
        - L3: "TP53: Tumor suppressor..."
        - L2 formatted: "TP53: Tumor suppressor..." → MATCH!
        """
        l3_label_lower = l3_label.lower().strip()

        for candidate in candidates:
            formatted_label = template.format(**candidate.dict())
            if formatted_label.lower().strip() == l3_label_lower:
                return candidate

        return None

    def calculate_stats(self, entities: List[List[L0Entity]]) -> dict:
        """Calculate pipeline statistics"""
        total = 0
        linked = 0
        unlinked = 0
        stage_counts = {"l1_only": 0, "l2_found": 0, "l3_linked": 0, "l3_only": 0}

        for text_entities in entities:
            for entity in text_entities:
                total += 1
                if entity.is_linked:
                    linked += 1
                else:
                    unlinked += 1
                stage_counts[entity.pipeline_stage] += 1

        return {
            "total_mentions": total,
            "linked": linked,
            "unlinked": unlinked,
            "linking_rate": linked / total if total > 0 else 0.0,
            "stages": stage_counts
        }
```

#### Pipeline Stages

- **`l1_only`**: Mention detected but no candidates found
- **`l2_found`**: Candidates found but not linked
- **`l3_linked`**: Successfully linked to entity
- **`l3_only`**: L3 found entity without L1 mention (loose mode only)

#### Strict vs Loose Matching

**Strict matching** (`strict_matching: true`):
- Only includes entities detected by L1
- L3 entities outside L1 mentions are ignored
- More conservative, fewer false positives

**Loose matching** (`strict_matching: false`):
- Includes all L3 entities, even if not in L1
- L3 may find additional entities L1 missed
- More comprehensive, potential false positives

#### Configuration Example

```yaml
- id: "l0"
  processor: "l0_aggregator"
  requires: ["l1", "l2", "l3"]
  inputs:
    l1_entities: {source: "l1_result", fields: "entities"}
    l2_candidates: {source: "l2_result", fields: "candidates"}
    l3_entities: {source: "l3_result", fields: "entities"}
  output: {key: "l0_result"}
  schema:
    template: "{label}: {description}"
  config:
    min_confidence: 0.0
    include_unlinked: true
    return_all_candidates: false
    strict_matching: true
    position_tolerance: 2
```

---

## Configuration System

### ConfigBuilder API

The `ConfigBuilder` provides a fluent API for creating pipeline configurations programmatically.

**ConfigBuilder** (src/glinker/core/builders.py)

```python
class ConfigBuilder:
    """
    Unified configuration builder with automatic defaults

    Simple usage (auto dict layer):
        builder = ConfigBuilder(name="demo")
        builder.l1.spacy(model="en_core_sci_sm")
        builder.l3.configure(model="...")
        config = builder.get_config()

    Advanced usage (custom layers):
        builder = ConfigBuilder(name="production")
        builder.l1.spacy(...)
        builder.l2.add("redis", priority=2, ttl=3600)
        builder.l2.add("postgres", priority=0)
        builder.l3.configure(...)
        builder.save("config.yaml")
    """

    def __init__(self, name: str = "pipeline", description: str = None)

    # L1 Builder
    class L1Builder:
        def spacy(
            self,
            model: str = "en_core_sci_sm",
            device: str = "cpu",
            batch_size: int = 32,
            max_right_context: int = 50,
            max_left_context: int = 50,
            min_entity_length: int = 2,
            include_noun_chunks: bool = False
        ) -> ConfigBuilder

        def gliner(
            self,
            model: str,
            labels: List[str],
            token: Optional[str] = None,
            device: str = "cpu",
            threshold: float = 0.3,
            flat_ner: bool = True,
            multi_label: bool = False,
            batch_size: int = 16,
            max_right_context: int = 50,
            max_left_context: int = 50,
            min_entity_length: int = 2,
            use_precomputed_embeddings: bool = False,
            max_length: Optional[int] = 512
        ) -> ConfigBuilder

    # L2 Builder
    class L2Builder:
        def add(
            self,
            layer_type: Literal["dict", "redis", "elasticsearch", "postgres"],
            priority: int = 0,
            write: bool = None,
            search_mode: List[str] = None,
            ttl: int = None,
            cache_policy: str = None,
            fuzzy_similarity: float = None,
            **db_config
        ) -> ConfigBuilder

        def embeddings(
            self,
            enabled: bool = True,
            model_name: str = "knowledgator/gliner-linker-large-v1.0",
            dim: int = 768,
            precompute_on_load: bool = False
        ) -> ConfigBuilder

    # L3 Builder
    class L3Builder:
        def configure(
            self,
            model: str = "knowledgator/gliner-linker-large-v1.0",
            token: Optional[str] = None,
            device: str = "cpu",
            threshold: float = 0.5,
            flat_ner: bool = True,
            multi_label: bool = False,
            batch_size: int = 1,
            use_precomputed_embeddings: bool = False,
            cache_embeddings: bool = False,
            max_length: Optional[int] = 512
        ) -> ConfigBuilder

    # L0 Builder
    class L0Builder:
        def configure(
            self,
            min_confidence: float = 0.0,
            include_unlinked: bool = True,
            return_all_candidates: bool = False,
            strict_matching: bool = True,
            position_tolerance: int = 2
        ) -> ConfigBuilder

    # Main methods
    def set_schema_template(self, template: str) -> ConfigBuilder
    def get_config(self) -> Dict[str, Any]
    def build(self) -> Dict[str, Any]
    def save(self, filepath: str) -> None
```

#### Usage Examples

**Simple configuration (auto dict layer):**

```python
from glinker.core.builders import ConfigBuilder

builder = ConfigBuilder(name="simple_pipeline")
builder.l1.spacy(model="en_core_sci_sm", batch_size=32)
builder.l3.configure(model="knowledgator/gliner-linker-large-v1.0")
builder.save("configs/simple.yaml")
```

**Production configuration (multiple layers):**

```python
builder = ConfigBuilder(name="production_pipeline")

# L1 with GLiNER
builder.l1.gliner(
    model="knowledgator/gliner-linker-large-v1.0",
    labels=["gene", "protein", "disease"],
    threshold=0.3
)

# L2 with full database stack
builder.l2.add("dict", priority=3, fuzzy_similarity=0.6)
builder.l2.add("redis", priority=2, ttl=3600, host="localhost", port=6379)
builder.l2.add("elasticsearch", priority=1, hosts=["http://localhost:9200"])
builder.l2.add("postgres", priority=0, database="entities_db", user="postgres")
builder.l2.embeddings(enabled=True, model_name="knowledgator/gliner-linker-large-v1.0")

# L3 with precomputed embeddings
builder.l3.configure(
    model="knowledgator/gliner-linker-large-v1.0",
    threshold=0.5,
    use_precomputed_embeddings=True
)

# L0 with strict matching
builder.l0.configure(
    min_confidence=0.3,
    strict_matching=True,
    position_tolerance=2
)

# Set schema template
builder.set_schema_template("{label}: {description}")

builder.save("configs/production.yaml")
```

### YAML Configuration

Complete pipeline configuration example:

```yaml
name: "entity_linking_pipeline"
description: "Four-layer entity linking with BiEncoder embeddings"

nodes:
  # L1: Named Entity Recognition
  - id: "l1"
    processor: "l1_spacy"
    inputs:
      texts:
        source: "$input"
        fields: "texts"
    output:
      key: "l1_result"
    config:
      model: "en_core_sci_sm"
      device: "cpu"
      batch_size: 32
      max_right_context: 50
      max_left_context: 50
      min_entity_length: 2
      include_noun_chunks: false

  # L2: Multi-Layer Database Search
  - id: "l2"
    processor: "l2_chain"
    requires: ["l1"]
    inputs:
      mentions:
        source: "l1_result"
        fields: "entities"
    output:
      key: "l2_result"
    schema:
      template: "{label}: {description}"
    config:
      max_candidates: 10
      min_popularity: 0
      layers:
        - type: "dict"
          priority: 3
          write: true
          search_mode: ["exact", "fuzzy"]
          ttl: 0
          cache_policy: "always"
          field_mapping:
            entity_id: "entity_id"
            label: "label"
            aliases: "aliases"
            description: "description"
            entity_type: "entity_type"
            popularity: "popularity"
            embedding: "embedding"
            embedding_model_id: "embedding_model_id"
          fuzzy:
            max_distance: 64
            min_similarity: 0.6
            n_gram_size: 3
            prefix_length: 1

        - type: "redis"
          priority: 2
          write: true
          search_mode: ["exact"]
          ttl: 3600
          cache_policy: "always"
          config:
            host: "localhost"
            port: 6379
            db: 0
          field_mapping:
            entity_id: "entity_id"
            label: "label"
            aliases: "aliases"
            description: "description"
            entity_type: "entity_type"
            popularity: "popularity"

        - type: "elasticsearch"
          priority: 1
          write: true
          search_mode: ["exact", "fuzzy"]
          ttl: 86400
          cache_policy: "miss"
          config:
            hosts: ["http://localhost:9200"]
            index_name: "entities"
          field_mapping:
            entity_id: "entity_id"
            label: "label"
            aliases: "aliases"
            description: "description"
            entity_type: "entity_type"
            popularity: "popularity"
          fuzzy:
            min_similarity: 0.3

        - type: "postgres"
          priority: 0
          write: false
          search_mode: ["exact", "fuzzy"]
          config:
            host: "localhost"
            port: 5432
            database: "entities_db"
            user: "postgres"
            password: "postgres"
          field_mapping:
            entity_id: "entity_id"
            label: "label"
            aliases: "aliases"
            description: "description"
            entity_type: "entity_type"
            popularity: "popularity"
          fuzzy:
            min_similarity: 0.3

      embeddings:
        enabled: true
        model_name: "knowledgator/gliner-linker-large-v1.0"
        dim: 768
        precompute_on_load: false

  # L3: Entity Disambiguation
  - id: "l3"
    processor: "l3_batch"
    requires: ["l2"]
    inputs:
      texts:
        source: "$input"
        fields: "texts"
      candidates:
        source: "l2_result"
        fields: "candidates"
    output:
      key: "l3_result"
    schema:
      template: "{label}: {description}"
    config:
      model_name: "knowledgator/gliner-linker-large-v1.0"
      device: "cpu"
      threshold: 0.5
      flat_ner: true
      multi_label: false
      batch_size: 1
      use_precomputed_embeddings: true
      cache_embeddings: false
      max_length: 512

  # L0: Final Aggregation
  - id: "l0"
    processor: "l0_aggregator"
    requires: ["l1", "l2", "l3"]
    inputs:
      l1_entities:
        source: "l1_result"
        fields: "entities"
      l2_candidates:
        source: "l2_result"
        fields: "candidates"
      l3_entities:
        source: "l3_result"
        fields: "entities"
    output:
      key: "l0_result"
    schema:
      template: "{label}: {description}"
    config:
      min_confidence: 0.0
      include_unlinked: true
      return_all_candidates: false
      strict_matching: true
      position_tolerance: 2
```

---

## Database Layers

### Setup

**PostgreSQL:**

```bash
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=entities_db \
  postgres:15-alpine
```

Schema:
```sql
CREATE TABLE entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    label VARCHAR(255) NOT NULL,
    description TEXT,
    entity_type VARCHAR(100),
    popularity INTEGER DEFAULT 0,
    embedding BYTEA,
    embedding_model_id VARCHAR(255)
);

CREATE TABLE aliases (
    entity_id VARCHAR(255) REFERENCES entities(entity_id) ON DELETE CASCADE,
    alias VARCHAR(255) NOT NULL,
    PRIMARY KEY (entity_id, alias)
);

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_entities_label_trgm ON entities USING gin(label gin_trgm_ops);
```

**Elasticsearch:**

```bash
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.11.0
```

Index mapping:
```json
{
  "mappings": {
    "properties": {
      "entity_id": {"type": "keyword"},
      "label": {"type": "text"},
      "aliases": {"type": "text"},
      "description": {"type": "text"},
      "entity_type": {"type": "keyword"},
      "popularity": {"type": "integer"},
      "embedding": {"type": "dense_vector", "dims": 768},
      "embedding_model_id": {"type": "keyword"}
    }
  }
}
```

**Redis:**

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### Loading Entities

**Entity JSONL format:**

```jsonl
{"entity_id": "Q12345", "label": "TP53", "aliases": ["p53", "tumor protein p53"], "description": "Tumor suppressor gene", "entity_type": "gene", "popularity": 1000000}
{"entity_id": "Q67890", "label": "BRCA1", "aliases": ["breast cancer 1"], "description": "DNA repair protein", "entity_type": "gene", "popularity": 800000}
```

**Load via DAGExecutor:**

```python
from glinker.core.factory import ProcessorFactory

executor = ProcessorFactory.create_pipeline("config.yaml", verbose=True)

# Load to all writable layers
executor.load_entities(
    filepath="entities.jsonl",
    target_layers=None,  # None = all writable layers
    batch_size=1000,
    overwrite=False
)

# Load to specific layers
executor.load_entities(
    filepath="entities.jsonl",
    target_layers=["dict", "redis"],
    batch_size=1000,
    overwrite=True
)
```

**Count entities:**

```python
counts = executor.count_entities()
# {'dict': 5000, 'redis': 5000, 'elasticsearch': 5000, 'postgres': 5000}
```

**Clear layers:**

```python
executor.clear_databases(layer_names=["redis", "dict"])
```

---

## Embedding System

### BiEncoder Support

GLiNKER supports **precomputed label embeddings** for BiEncoder GLiNER models, providing **100x speedup** for entity disambiguation.

#### How It Works

1. **L2** stores entities with `embedding` and `embedding_model_id` fields
2. **L3** checks if candidates have precomputed embeddings matching current model
3. If available, uses `model.predict_with_embeds()` instead of computing on-the-fly

#### Configuration

**L2 with embeddings:**

```yaml
- id: "l2"
  schema:
    template: "{label}: {description}"
  config:
    embeddings:
      enabled: true
      model_name: "knowledgator/gliner-linker-large-v1.0"
      dim: 768
      precompute_on_load: false
```

**L3 using precomputed embeddings:**

```yaml
- id: "l3"
  schema:
    template: "{label}: {description}"  # MUST match L2 template
  config:
    model_name: "knowledgator/gliner-linker-large-v1.0"
    use_precomputed_embeddings: true
    cache_embeddings: false
```

#### Precomputing Embeddings

**Method 1: Via DAGExecutor**

```python
from glinker.core.factory import ProcessorFactory

executor = ProcessorFactory.create_pipeline("config.yaml", verbose=True)

# Load entities first
executor.load_entities("entities.jsonl", target_layers=["dict", "postgres"])

# Precompute embeddings
results = executor.precompute_embeddings(
    target_layers=["dict", "postgres"],  # Layers to update
    batch_size=64
)
# {'dict': 5000, 'postgres': 5000}
```

**Method 2: Programmatic**

```python
# Get L2 and L3 processors
l2_processor = executor.processors['l2']
l3_processor = executor.processors['l3']

# Create encoder function
def encoder_fn(labels):
    return l3_processor.component.encode_labels(labels, batch_size=64)

# Precompute
template = "{label}: {description}"
model_id = "knowledgator/gliner-linker-large-v1.0"

results = l2_processor.component.precompute_embeddings(
    encoder_fn=encoder_fn,
    template=template,
    model_id=model_id,
    target_layers=["dict"],
    batch_size=64
)
```

#### Template Consistency

**CRITICAL**: L2 and L3 must use the **same template** for embeddings to work:

```yaml
# L2 template
- id: "l2"
  schema:
    template: "{label}: {description}"

# L3 template (MUST MATCH!)
- id: "l3"
  schema:
    template: "{label}: {description}"
```

#### Performance

Without precomputed embeddings:
- 1000 candidates × 10 texts = **10 seconds**

With precomputed embeddings:
- 1000 candidates × 10 texts = **0.1 seconds**

Speedup: **100x**

#### Cache Write-Back

L3 can cache newly computed embeddings back to L2:

```yaml
- id: "l3"
  config:
    use_precomputed_embeddings: true
    cache_embeddings: true  # Enable cache write-back
```

When `cache_embeddings: true`:
- If candidate has no embedding → compute and store in L2
- Next time → use precomputed embedding

---

## Field Resolution

### Input Configuration

**InputConfig** (src/glinker/core/dag.py)
```python
class InputConfig(BaseModel):
    source: str              # "l1_result", "outputs[-1]", "$input"
    fields: Union[str, List[str], None] = None  # "entities[*].text"
    reduce: Literal["all", "first", "last", "flatten"] = "all"
    reshape: Optional[ReshapeConfig] = None
    template: Optional[str] = None  # "{label}: {description}"
    filter: Optional[str] = None    # "score > 0.5"
    default: Any = None
```

### Field Path Syntax

Supports JSONPath-like syntax:

- **`entities`** → Field access
- **`entities[*]`** → Iterate list
- **`entities[*][*].text`** → Nested iteration
- **`entities[0]`** → Index access
- **`entities[1:]`** → Slicing
- **`entities[*].text`** → Extract field from each item

### Examples

**Extract entity texts:**

```yaml
inputs:
  mentions:
    source: "l1_result"
    fields: "entities[*][*].text"  # Nested list → flat list of texts
    reduce: "flatten"
```

**Get last output:**

```yaml
inputs:
  data:
    source: "outputs[-1]"
```

**Format with template:**

```yaml
inputs:
  labels:
    source: "l2_result"
    fields: "candidates"
    template: "{label}: {description}"
```

**Filter by score:**

```yaml
inputs:
  high_conf:
    source: "l3_result"
    fields: "entities"
    filter: "score > 0.7"
```

**Reduce modes:**

```yaml
# Get first item
inputs:
  first: {source: "l1_result", fields: "entities", reduce: "first"}

# Get last item
inputs:
  last: {source: "l1_result", fields: "entities", reduce: "last"}

# Flatten nested lists
inputs:
  flat: {source: "l1_result", fields: "entities[*][*]", reduce: "flatten"}
```

---

## API Reference

### Core Classes

#### DAGExecutor

```python
class DAGExecutor:
    def __init__(self, pipeline: DAGPipeline, verbose: bool = False)

    def execute(self, pipeline_input: Any) -> PipeContext:
        """Execute full pipeline and return context with all outputs"""

    def load_entities(
        self,
        filepath: str,
        target_layers: List[str] = None,
        batch_size: int = 1000,
        overwrite: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """Load entities from JSONL file into database layers"""

    def precompute_embeddings(
        self,
        target_layers: List[str] = None,
        batch_size: int = 32
    ) -> Dict[str, int]:
        """Precompute embeddings for all entities using L3 model"""

    def clear_databases(self, layer_names: List[str] = None) -> Dict[str, bool]:
        """Clear database layers"""

    def count_entities(self) -> Dict[str, Dict[str, int]]:
        """Count entities in all database layers"""
```

#### PipeContext

```python
class PipeContext:
    def __init__(self, pipeline_input: Any = None)

    def set(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store output with optional metadata"""

    def get(self, source: str) -> Any:
        """Get data by key, index, or $input"""

    def has(self, key: str) -> bool:
        """Check if output exists"""

    def get_all_outputs(self) -> Dict[str, Any]:
        """Get all outputs as dict"""

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for output"""

    def get_execution_order(self) -> List[str]:
        """Get execution order of outputs"""

    def to_json(self, filepath: str = None, indent: int = 2) -> str:
        """Serialize to JSON"""

    @classmethod
    def from_json(cls, json_data: str = None, filepath: str = None) -> PipeContext:
        """Deserialize from JSON"""
```

#### ProcessorFactory

```python
class ProcessorFactory:
    @staticmethod
    def create_pipeline(config_path: str, verbose: bool = False) -> DAGExecutor:
        """Create DAG pipeline from YAML config"""

    @staticmethod
    def create_from_dict(config_dict: dict, verbose: bool = False) -> DAGExecutor:
        """Create pipeline from dict (programmatic use)"""
```

#### ConfigBuilder

```python
class ConfigBuilder:
    def __init__(self, name: str = "pipeline", description: str = None)

    # Sub-builders
    l1: L1Builder
    l2: L2Builder
    l3: L3Builder
    l0: L0Builder

    def set_schema_template(self, template: str) -> ConfigBuilder
    def get_config(self) -> Dict[str, Any]
    def build(self) -> Dict[str, Any]
    def save(self, filepath: str) -> None
```

### Layer APIs

#### L1

```python
class L1SpacyComponent:
    def extract_entities(self, text: str) -> list[L1Entity]
    def filter_by_length(self, entities, min_length) -> list[L1Entity]
    def deduplicate(self, entities) -> list[L1Entity]
    def sort_by_position(self, entities) -> list[L1Entity]
    def add_noun_chunks(self, text, entities) -> list[L1Entity]

class L1GlinerComponent:
    def extract_entities(self, text: str) -> List[L1Entity]
    def encode_labels(self, labels: List[str], batch_size: int) -> torch.Tensor
    def filter_by_length(self, entities, min_length) -> List[L1Entity]
    def deduplicate(self, entities) -> List[L1Entity]
    def sort_by_position(self, entities) -> List[L1Entity]

    @property
    def supports_precomputed_embeddings(self) -> bool

class L1SpacyProcessor:
    def __call__(self, texts: List[str] = None) -> L1Output

class L1GlinerProcessor:
    def __call__(self, texts: List[str] = None) -> L1Output
```

#### L2

```python
class DatabaseChainComponent:
    def search(self, mention: str) -> List[DatabaseRecord]
    def filter_by_popularity(self, records, min_popularity) -> List[DatabaseRecord]
    def deduplicate_candidates(self, records) -> List[DatabaseRecord]
    def limit_candidates(self, records, limit) -> List[DatabaseRecord]
    def sort_by_popularity(self, records) -> List[DatabaseRecord]

    def load_entities(self, filepath, target_layers, ...) -> Dict[str, int]
    def clear_layers(self, layer_names: List[str])
    def count_entities(self) -> Dict[str, int]
    def precompute_embeddings(...) -> Dict[str, int]

class L2Processor:
    def __call__(self, mentions: Union[List[str], List[List[Any]]]) -> L2Output
    def format_label(self, record: DatabaseRecord) -> str
    def precompute_embeddings(...) -> Dict[str, int]
```

#### L3

```python
class L3Component:
    def predict_entities(self, text: str, labels: List[str]) -> List[L3Entity]
    def predict_with_embeddings(self, text, labels, embeddings) -> List[L3Entity]
    def encode_labels(self, labels: List[str], batch_size: int) -> torch.Tensor
    def filter_by_score(self, entities, threshold) -> List[L3Entity]
    def sort_by_position(self, entities) -> List[L3Entity]
    def deduplicate_entities(self, entities) -> List[L3Entity]

    @property
    def supports_precomputed_embeddings(self) -> bool

class L3Processor:
    def __call__(self, texts: List[str], candidates: List[List[Any]]) -> L3Output
```

#### L0

```python
class L0Component:
    def aggregate(
        self,
        l1_entities: List[List[L1Entity]],
        l2_candidates: List[List[DatabaseRecord]],
        l3_entities: List[List[L3Entity]],
        template: str
    ) -> List[List[L0Entity]]

    def filter_by_confidence(self, entities, min_confidence) -> List[List[L0Entity]]
    def sort_by_confidence(self, entities) -> List[List[L0Entity]]
    def calculate_stats(self, entities) -> dict

class L0Processor:
    def __call__(
        self,
        l1_entities: List[List[L1Entity]],
        l2_candidates: List[List[DatabaseRecord]],
        l3_entities: List[List[L3Entity]]
    ) -> L0Output
```

---

## Complete Examples

### Example 1: Simple Pipeline

```python
from glinker.core.builders import ConfigBuilder
from glinker.core.factory import ProcessorFactory
from glinker.l1.models import L1Input

# Build configuration
builder = ConfigBuilder(name="simple_pipeline")
builder.l1.spacy(model="en_core_sci_sm", batch_size=32)
builder.l3.configure(model="knowledgator/gliner-linker-large-v1.0", threshold=0.5)
builder.set_schema_template("{label}")
builder.save("simple.yaml")

# Create executor
executor = ProcessorFactory.create_pipeline("simple.yaml", verbose=True)

# Load entities (auto dict layer)
executor.load_entities("entities.jsonl")

# Execute
input_data = L1Input(texts=[
    "TP53 mutations are found in many cancers.",
    "BRCA1 is involved in DNA repair."
])

result = executor.execute(input_data)

# Access results
l0_output = result.get("l0_result")
print(f"Linked {l0_output.stats['linked']} / {l0_output.stats['total_mentions']} entities")

for text_entities in l0_output.entities:
    for entity in text_entities:
        if entity.is_linked:
            print(f"{entity.mention_text} → {entity.linked_entity.label} ({entity.linked_entity.confidence:.2f})")
```

### Example 2: Production Pipeline with Precomputed Embeddings

```python
from glinker.core.builders import ConfigBuilder
from glinker.core.factory import ProcessorFactory

# Build configuration
builder = ConfigBuilder(name="production")

builder.l1.spacy(model="en_core_sci_sm", batch_size=64)

builder.l2.add("dict", priority=3, fuzzy_similarity=0.6)
builder.l2.add("redis", priority=2, ttl=3600, host="localhost", port=6379)
builder.l2.add("postgres", priority=0, database="entities_db", user="postgres", password="postgres")
builder.l2.embeddings(
    enabled=True,
    model_name="knowledgator/gliner-linker-large-v1.0",
    dim=768
)

builder.l3.configure(
    model="knowledgator/gliner-linker-large-v1.0",
    threshold=0.5,
    use_precomputed_embeddings=True,
    batch_size=1
)

builder.l0.configure(
    min_confidence=0.3,
    strict_matching=True,
    position_tolerance=2
)

builder.set_schema_template("{label}: {description}")
builder.save("production.yaml")

# Create executor
executor = ProcessorFactory.create_pipeline("production.yaml", verbose=True)

# Load entities
executor.load_entities(
    filepath="entities.jsonl",
    target_layers=["dict", "redis", "postgres"],
    batch_size=1000,
    overwrite=False
)

# Precompute embeddings
executor.precompute_embeddings(
    target_layers=["dict", "postgres"],
    batch_size=64
)

# Execute (100x faster with precomputed embeddings!)
from glinker.l1.models import L1Input
input_data = L1Input(texts=["TP53 is a tumor suppressor gene."])
result = executor.execute(input_data)

# Get aggregated output
l0_output = result.get("l0_result")
print(l0_output.stats)
```

### Example 3: Programmatic Configuration

```python
from glinker.core.dag import DAGPipeline, PipeNode, InputConfig, OutputConfig
from glinker.core.factory import ProcessorFactory

# Build config dict
config = {
    "name": "custom_pipeline",
    "nodes": [
        {
            "id": "l1",
            "processor": "l1_spacy",
            "inputs": {
                "texts": {"source": "$input", "fields": "texts"}
            },
            "output": {"key": "l1_result"},
            "config": {
                "model": "en_core_sci_sm",
                "batch_size": 32,
                "min_entity_length": 3
            }
        },
        {
            "id": "l2",
            "processor": "l2_chain",
            "requires": ["l1"],
            "inputs": {
                "mentions": {"source": "l1_result", "fields": "entities"}
            },
            "output": {"key": "l2_result"},
            "schema": {"template": "{label}"},
            "config": {
                "max_candidates": 10,
                "layers": [
                    {
                        "type": "dict",
                        "priority": 0,
                        "write": True,
                        "search_mode": ["exact", "fuzzy"],
                        "ttl": 0,
                        "cache_policy": "always",
                        "fuzzy": {"min_similarity": 0.6}
                    }
                ]
            }
        },
        {
            "id": "l3",
            "processor": "l3_batch",
            "requires": ["l2"],
            "inputs": {
                "texts": {"source": "$input", "fields": "texts"},
                "candidates": {"source": "l2_result", "fields": "candidates"}
            },
            "output": {"key": "l3_result"},
            "schema": {"template": "{label}"},
            "config": {
                "model_name": "knowledgator/gliner-linker-large-v1.0",
                "threshold": 0.5
            }
        },
        {
            "id": "l0",
            "processor": "l0_aggregator",
            "requires": ["l1", "l2", "l3"],
            "inputs": {
                "l1_entities": {"source": "l1_result", "fields": "entities"},
                "l2_candidates": {"source": "l2_result", "fields": "candidates"},
                "l3_entities": {"source": "l3_result", "fields": "entities"}
            },
            "output": {"key": "l0_result"},
            "config": {
                "min_confidence": 0.0,
                "strict_matching": True
            }
        }
    ]
}

# Create executor from dict
executor = ProcessorFactory.create_from_dict(config, verbose=True)

# Execute
from glinker.l1.models import L1Input
result = executor.execute(L1Input(texts=["Example text"]))
```

### Example 4: Context Debugging

```python
from glinker.core.factory import ProcessorFactory
from glinker.l1.models import L1Input

executor = ProcessorFactory.create_pipeline("config.yaml")

# Execute
input_data = L1Input(texts=["TP53 mutations"])
context = executor.execute(input_data)

# Save context for debugging
context.to_json("debug_context.json")

# Later: restore and inspect
from glinker.core.dag import PipeContext
restored = PipeContext.from_json(filepath="debug_context.json")

# Inspect all outputs
for key in restored.get_execution_order():
    output = restored.get(key)
    metadata = restored.get_metadata(key)
    print(f"{key}: {type(output)} - {metadata}")

# Get specific outputs
l1_result = restored.get("l1_result")
l2_result = restored.get("l2_result")
l3_result = restored.get("l3_result")
l0_result = restored.get("l0_result")
```

### Example 5: Batch Processing

```python
from glinker.core.factory import ProcessorFactory
from glinker.l1.models import L1Input
import json

executor = ProcessorFactory.create_pipeline("config.yaml", verbose=True)
executor.load_entities("entities.jsonl")

# Load texts from file
with open("texts.jsonl", "r") as f:
    texts = [json.loads(line)["text"] for line in f]

# Process in batches
batch_size = 100
all_results = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    input_data = L1Input(texts=batch)

    context = executor.execute(input_data)
    l0_output = context.get("l0_result")

    all_results.append(l0_output)

    print(f"Batch {i//batch_size + 1}: {l0_output.stats['linking_rate']:.2%} linking rate")

# Aggregate statistics
total_mentions = sum(r.stats["total_mentions"] for r in all_results)
total_linked = sum(r.stats["linked"] for r in all_results)
print(f"Overall: {total_linked}/{total_mentions} ({total_linked/total_mentions:.2%})")
```

---

## Design Patterns Summary

### 1. Component-Processor Separation
- **Components**: Pure, stateless methods
- **Processors**: Stateful orchestration with configurable pipelines
- **Benefit**: Easy testing, flexible composition

### 2. Factory + Registry
- **Registry**: Auto-registration via decorators
- **Factory**: Creates pipelines from configs
- **Benefit**: Plugin architecture, clean DI

### 3. DAG Execution
- **Declarative**: Define in YAML
- **Automatic**: Dependency resolution from inputs
- **Parallel**: Topological sort with level grouping
- **Benefit**: Complex pipelines without manual orchestration

### 4. Unified Data Models
- **Pydantic**: Type-safe, validated
- **Hierarchical**: L1Entity → DatabaseRecord → L3Entity → L0Entity
- **Serializable**: JSON export/import
- **Benefit**: Clear contracts, easy debugging

### 5. Cache Hierarchy
- **Multi-layer**: Dict → Redis → ES → Postgres
- **Automatic write-back**: Upper layers cached on miss
- **Configurable policies**: always/miss/hit
- **Benefit**: Fast lookups, reduced database load

### 6. Precomputed Embeddings
- **BiEncoder support**: 100x speedup
- **Template consistency**: Same format across L2/L3
- **Lazy computation**: Compute on demand, cache for reuse
- **Benefit**: Production-ready performance

---
