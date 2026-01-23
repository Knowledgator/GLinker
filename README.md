# EntityLinker

A modular, DAG-based entity linking framework that combines Named Entity Recognition (NER), multi-layer database search, and neural entity disambiguation into a unified pipeline.

## Table of Contents

- [Philosophy](#philosophy)
- [Architecture Overview](#architecture-overview)
- [The Four Layers](#the-four-layers)
  - [L1: Named Entity Recognition](#l1-named-entity-recognition)
  - [L2: Candidate Generation](#l2-candidate-generation)
  - [L3: Entity Disambiguation](#l3-entity-disambiguation)
  - [L0: Aggregation Layer](#l0-aggregation-layer)
- [DAG Pipeline System](#dag-pipeline-system)
  - [Pipeline Configuration](#pipeline-configuration)
  - [Input/Output Resolution](#inputoutput-resolution)
  - [Field Extraction Syntax](#field-extraction-syntax)
- [Embedding Management](#embedding-management)
  - [Precomputed Embeddings](#precomputed-embeddings)
  - [On-the-fly Caching](#on-the-fly-caching)
- [Database Layers](#database-layers)
  - [Layer Hierarchy](#layer-hierarchy)
  - [Cache Policies](#cache-policies)
  - [Fuzzy Search](#fuzzy-search)
- [Configuration Reference](#configuration-reference)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)

---

## Philosophy

EntityLinker is built on several core principles:

### 1. Separation of Concerns

Each processing stage (mention extraction, candidate search, disambiguation, aggregation) is isolated into its own layer. This allows:
- Independent optimization of each stage
- Easy replacement of components (e.g., swap spaCy for a custom NER)
- Clear debugging boundaries

### 2. Declarative Configuration

Pipelines are defined in YAML, not code. This enables:
- Version-controlled pipeline definitions
- Easy experimentation with different configurations
- Non-programmer access to pipeline tuning

### 3. Component-Processor Pattern

Every layer follows the same pattern:
- **Component**: Implements discrete, stateless methods (`search`, `predict`, `filter`)
- **Processor**: Orchestrates component methods via configurable pipelines
- **Config**: Pydantic model for validated configuration

```
┌─────────────────────────────────────────────────┐
│                   Processor                      │
│  ┌─────────────────────────────────────────┐    │
│  │              Component                   │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │ method1 │ │ method2 │ │ method3 │   │    │
│  │  └─────────┘ └─────────┘ └─────────┘   │    │
│  └─────────────────────────────────────────┘    │
│                      ↑                           │
│                   Config                         │
└─────────────────────────────────────────────────┘
```

### 4. Caching as a First-Class Citizen

The framework treats caching not as an afterthought but as a core feature:
- Multi-layer cache hierarchy (memory → Redis → Elasticsearch → PostgreSQL)
- Automatic cache write-back on cache miss
- Embedding precomputation and on-the-fly caching

### 5. Schema Consistency

A single template (e.g., `"{label}: {description}"`) is used across L2, L3, and L0 to ensure consistent label formatting throughout the pipeline.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            DAG Executor                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         Pipeline Context                            │  │
│  │   $input → L1 → l1_result → L2 → l2_result → L3 → l3_result → L0   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │    L1    │   │    L2    │   │    L3    │   │    L0    │             │
│  │  spaCy   │ → │ Database │ → │  GLiNER  │ → │Aggregator│             │
│  │   NER    │   │  Chain   │   │  Linker  │   │          │             │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘             │
│       ↓              ↓              ↓              ↓                    │
│   mentions      candidates      entities      final output              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The Four Layers

### L1: Named Entity Recognition

**Purpose**: Extract entity mentions from raw text.

**Component**: `L1Component` (spaCy-based)
**Processor**: `L1BatchProcessor`

**Input**: Raw text strings
**Output**: List of `L1Entity` with:
- `text`: The mention text (e.g., "TP53")
- `start`, `end`: Character positions in source text
- `left_context`, `right_context`: Surrounding text for disambiguation
- `label`: Entity type from NER model

**Configuration**:
```yaml
config:
  model: "en_core_sci_sm"      # spaCy model name
  device: "cpu"                 # cpu or cuda
  batch_size: 32                # Batch size for processing
  max_left_context: 50          # Characters of left context to capture
  max_right_context: 50         # Characters of right context to capture
  min_entity_length: 2          # Minimum mention length
  include_noun_chunks: true     # Also extract noun chunks as mentions
```

**Models**:
- `en_core_sci_sm`: Scientific/biomedical text (scispaCy)
- `en_core_web_sm`: General English text

---

### L2: Candidate Generation

**Purpose**: Find candidate entities from knowledge bases for each mention.

**Component**: `DatabaseChainComponent` (multi-layer database search)
**Processor**: `L2ChainProcessor`

**Input**: List of mention texts from L1
**Output**: List of `DatabaseRecord` candidates with:
- `entity_id`: Unique identifier (e.g., "MESH:D001943")
- `label`: Canonical entity name
- `description`: Entity description
- `aliases`: Alternative names
- `entity_type`: Category (gene, disease, drug, etc.)
- `popularity`: Usage frequency score
- `embedding`: Precomputed label embedding (optional)
- `embedding_model_id`: Model used for embedding

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                  DatabaseChainComponent                      │
│                                                              │
│   Query: "TP53"                                              │
│        ↓                                                     │
│   ┌─────────┐   miss   ┌──────────────┐   miss   ┌────────┐ │
│   │  Dict   │ ───────→ │ Elasticsearch │ ───────→ │Postgres│ │
│   │(memory) │          │   (search)    │          │ (SQL)  │ │
│   └────┬────┘          └──────┬────────┘          └───┬────┘ │
│        │ hit                  │ hit                   │ hit  │
│        ↓                      ↓                       ↓      │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Cache Write-Back                        │   │
│   │   Results from lower layers written to upper layers  │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Configuration**:
```yaml
config:
  max_candidates: 5           # Max candidates per mention
  min_popularity: 0           # Minimum popularity threshold
  embeddings:
    enabled: true
    model_name: "BioMike/gliner-deberta-base-v1-post"
    dim: 768
  layers:
    - type: "dict"            # In-memory dictionary
      priority: 0             # Lower = checked first
      search_mode: ["exact", "fuzzy"]
      write: true             # Can write cache entries
      cache_policy: "always"
      ttl: 0                  # 0 = no expiration
      fuzzy:
        max_distance: 64
        min_similarity: 0.6
        n_gram_size: 3
        prefix_length: 1
      field_mapping:
        entity_id: "entity_id"
        label: "label"
        # ... maps standard fields to layer-specific fields
```

**Search Modes**:
- `exact`: Exact string match on label and aliases
- `fuzzy`: Similarity-based search (Levenshtein, n-gram, trigram)

---

### L3: Entity Disambiguation

**Purpose**: Select the correct entity from candidates using neural matching.

**Component**: `L3Component` (GLiNER-based)
**Processor**: `L3BatchProcessor`

**Input**:
- Original texts
- Candidates from L2

**Output**: List of `L3Entity` with:
- `text`: Matched text span
- `label`: Selected entity label (formatted with template)
- `start`, `end`: Character positions
- `score`: Confidence score (0-1)

**How It Works**:
```
Text: "TP53 mutations cause breast cancer"
Candidates: ["TP53: Tumor protein p53", "BRCA1: Breast cancer gene", ...]

GLiNER predicts:
  - "TP53" (pos 0-4) → "TP53: Tumor protein p53" (score: 0.92)
  - "breast cancer" (pos 21-34) → matched candidate (score: 0.87)
```

**Configuration**:
```yaml
config:
  model_name: "BioMike/gliner-deberta-base-v1-post"
  huggingface_token: "hf_..."
  device: "cuda"              # GPU recommended
  threshold: 0.5              # Minimum confidence score
  flat_ner: true              # Non-overlapping entities
  multi_label: false          # Single label per span
  batch_size: 1
  use_precomputed_embeddings: true   # Use cached embeddings
  cache_embeddings: true             # Cache new embeddings
schema:
  template: "{label}: {description}"  # Label format for GLiNER
```

**Embedding Modes**:

1. **No caching** (`use_precomputed_embeddings: false`, `cache_embeddings: false`):
   - Computes embeddings for every query
   - Slowest, but no storage needed

2. **Precomputed** (`use_precomputed_embeddings: true`, `cache_embeddings: false`):
   - Uses embeddings stored in L2 database
   - Requires running `executor.precompute_embeddings()` beforehand
   - Fastest inference

3. **On-the-fly caching** (`use_precomputed_embeddings: true`, `cache_embeddings: true`):
   - First query: compute and cache
   - Subsequent queries: use cached
   - Good balance of speed and flexibility

---

### L0: Aggregation Layer

**Purpose**: Combine outputs from L1, L2, L3 into a unified result with statistics.

**Component**: `L0Component`
**Processor**: `L0Processor`

**Input**:
- L1 entities (mentions)
- L2 candidates
- L3 linked entities

**Output**: `L0Output` with:
- `entities`: List of `L0Entity` per text
- `stats`: Pipeline statistics

**L0Entity Structure**:
```python
L0Entity(
    # From L1 - mention detection
    mention_text="TP53",
    mention_start=0,
    mention_end=4,
    left_context="",
    right_context=" mutations cause...",

    # From L2 - candidates
    candidates=[DatabaseRecord(...), ...],
    num_candidates=3,

    # From L3 - linking result
    linked_entity=LinkedEntity(
        entity_id="MESH:D001943",
        label="TP53",
        confidence=0.92,
        ...
    ),
    is_linked=True,

    # Aggregated
    pipeline_stage="l3_linked"  # l1_only | l2_found | l3_linked | l3_only
)
```

**Configuration**:
```yaml
config:
  min_confidence: 0.0         # Minimum linking confidence
  include_unlinked: true      # Include mentions without links
  return_all_candidates: true # Return all L2 candidates
  strict_matching: true       # Only L1→L3 matches (see below)
  position_tolerance: 2       # Fuzzy position matching (chars)
schema:
  template: "{label}: {description}"
```

**Matching Modes**:

`strict_matching: true` (default):
```
L1 mentions:  [TP53, BRCA1, cancer]
L3 entities:  [TP53, BRCA1, p53, tumor]
                            ↑ ignored (no L1 mention)
Result:       [TP53✓, BRCA1✓, cancer✗]
```

`strict_matching: false` (loose mode):
```
L1 mentions:  [TP53, BRCA1, cancer]
L3 entities:  [TP53, BRCA1, p53, tumor]
                            ↑ included as "l3_only"
Result:       [TP53✓, BRCA1✓, cancer✗, p53✓, tumor✓]
```

**Position Tolerance**:

L1 (spaCy) and L3 (GLiNER) may detect slightly different span boundaries:
```
L1: "TP53" at positions (10, 14)
L3: "TP53" at positions (10, 15)  ← includes trailing space

With position_tolerance=2: These match! (difference ≤ 2 chars)
```

**Pipeline Stages**:
- `l1_only`: Mention found, no candidates
- `l2_found`: Candidates found, not linked
- `l3_linked`: Successfully linked via L1→L2→L3 flow
- `l3_only`: GLiNER found entity not in L1 (loose mode only)

---

## DAG Pipeline System

### Pipeline Configuration

Pipelines are defined as Directed Acyclic Graphs (DAGs) in YAML:

```yaml
name: "entity_linking"
description: "Full entity linking pipeline"

nodes:
  - id: "l1"
    processor: "l1_batch"
    inputs:
      texts: {source: "$input", fields: "texts"}
    output: {key: "l1_result"}
    config: {...}

  - id: "l2"
    processor: "l2_chain"
    requires: ["l1"]           # Explicit dependency (optional)
    inputs:
      mentions: {source: "l1_result", fields: "entities"}
    output: {key: "l2_result"}
    config: {...}

  - id: "l3"
    processor: "l3_batch"
    inputs:
      texts: {source: "$input", fields: "texts"}
      candidates: {source: "l2_result", fields: "candidates"}
    output: {key: "l3_result"}
    config: {...}

  - id: "l0"
    processor: "l0_aggregator"
    requires: ["l1", "l2", "l3"]
    inputs:
      l1_entities: {source: "l1_result", fields: "entities"}
      l2_candidates: {source: "l2_result", fields: "candidates"}
      l3_entities: {source: "l3_result", fields: "entities"}
    output: {key: "l0_result"}
    config: {...}
```

### Input/Output Resolution

The DAG executor resolves inputs using special sources:

| Source | Description |
|--------|-------------|
| `$input` | Original input to pipeline |
| `node_id_result` | Output from specific node |
| `outputs[-1]` | Previous node's output |

### Field Extraction Syntax

Fields are extracted using JSONPath-like syntax:

| Syntax | Description | Example |
|--------|-------------|---------|
| `field` | Direct access | `texts` → `["text1", "text2"]` |
| `field[*]` | Iterate list | `entities[*]` → all entities |
| `field[*][*]` | Nested iteration | `entities[*][*].text` |
| `field[0]` | Index access | `entities[0]` → first entity |
| `field[1:]` | Slice | `entities[1:]` → skip first |

**Reduce Operations**:
```yaml
inputs:
  mentions:
    source: "l1_result"
    fields: "entities[*][*].text"
    reduce: "flatten"  # Flatten nested lists
```

### Execution Flow

```python
# 1. Create executor (initializes all processors once)
executor = ProcessorFactory.create_from_dict(yaml_config)

# 2. Load entities into L2 layers
executor.load_entities("entities.jsonl", target_layers=["dict"])

# 3. Optional: Setup embedding caching
executor.setup_l3_cache_writeback()

# 4. Execute pipeline
result = executor.execute({"texts": ["TP53 causes cancer"]})

# 5. Access results
l0_output = result.get("l0_result")
for entity in l0_output.entities[0]:
    print(f"{entity.mention_text} → {entity.linked_entity.label}")
```

---

## Embedding Management

### Precomputed Embeddings

For maximum performance, precompute embeddings offline:

```python
# 1. Load config with embeddings enabled
executor = ProcessorFactory.create_from_dict(yaml_config)

# 2. Load entities
executor.load_entities("entities.jsonl", target_layers=["dict"])

# 3. Precompute embeddings using L3 model
stats = executor.precompute_embeddings(
    target_layers=["dict"],  # Which layers to update
    batch_size=32
)
print(f"Computed {stats['total']} embeddings")

# 4. Now queries use precomputed embeddings
result = executor.execute({"texts": ["..."]})
```

**How it works**:
1. L2 retrieves all entities from target layers
2. Labels are formatted using schema template: `"{label}: {description}"`
3. L3 model encodes labels to embeddings
4. Embeddings are stored back in L2 layers

### On-the-fly Caching

For incremental processing without upfront precomputation:

```python
# 1. Create executor
executor = ProcessorFactory.create_from_dict(yaml_config)

# 2. Load entities (no embeddings yet)
executor.load_entities("entities.jsonl", target_layers=["dict"])

# 3. Enable cache writeback
executor.setup_l3_cache_writeback()

# 4. First query: computes and caches embeddings
result = executor.execute({"texts": ["TP53..."]})  # Slower

# 5. Subsequent queries: uses cached embeddings
result = executor.execute({"texts": ["TP53..."]})  # Faster!
```

**Config for on-the-fly caching**:
```yaml
# L2 config
embeddings:
  enabled: true
  model_name: "BioMike/gliner-deberta-base-v1-post"
  dim: 768

# L3 config
use_precomputed_embeddings: true
cache_embeddings: true
```

---

## Database Layers

### Layer Hierarchy

| Layer | Type | Speed | Capacity | Search | Use Case |
|-------|------|-------|----------|--------|----------|
| Dict | Memory | Fastest | <10K | Exact, Fuzzy | Development, small KBs |
| Redis | Cache | Fast | ~1M | Exact only | Production cache |
| Elasticsearch | Search | Medium | Unlimited | Full-text, Fuzzy | Primary search |
| PostgreSQL | SQL | Slower | Unlimited | SQL, Trigram | Persistent storage |

### Layer Configuration

```yaml
layers:
  - type: "dict"
    priority: 0              # Checked first
    search_mode: ["exact", "fuzzy"]
    write: true
    cache_policy: "always"
    ttl: 0                   # No expiration
    field_mapping:
      entity_id: "entity_id"
      label: "label"
      aliases: "aliases"
      description: "description"
      entity_type: "entity_type"
      popularity: "popularity"
    fuzzy:
      max_distance: 64
      min_similarity: 0.6
      n_gram_size: 3
      prefix_length: 1

  - type: "redis"
    priority: 1
    search_mode: ["exact"]
    config:
      host: "localhost"
      port: 6379
      db: 0
    write: true
    cache_policy: "miss"
    ttl: 3600                # 1 hour

  - type: "elasticsearch"
    priority: 2
    search_mode: ["exact", "fuzzy"]
    config:
      hosts: ["http://localhost:9200"]
      index_name: "entities"
    write: true
    cache_policy: "miss"
    ttl: 86400               # 24 hours
    fuzzy:
      max_distance: 2
      min_similarity: 0.3
      prefix_length: 1

  - type: "postgres"
    priority: 3
    search_mode: ["fuzzy"]
    config:
      host: "localhost"
      port: 5432
      database: "entities_db"
      user: "postgres"
      password: "postgres"
    write: false             # Read-only (source of truth)
    fuzzy:
      min_similarity: 0.3
```

### Cache Policies

| Policy | Behavior |
|--------|----------|
| `always` | Write to cache on every search |
| `miss` | Write only if not already cached |
| `hit` | Write only if already cached (refresh TTL) |

### Fuzzy Search

**Dict Layer** (n-gram based):
```yaml
fuzzy:
  max_distance: 64        # Maximum edit distance
  min_similarity: 0.6     # Minimum similarity (0-1)
  n_gram_size: 3          # Size of n-grams
  prefix_length: 1        # Required prefix match
```

**Elasticsearch** (Levenshtein):
```yaml
fuzzy:
  max_distance: 2         # Levenshtein distance (0-2)
  min_similarity: 0.3     # Minimum score
  prefix_length: 1        # Required prefix match
```

**PostgreSQL** (pg_trgm):
```yaml
fuzzy:
  min_similarity: 0.3     # Trigram similarity threshold
```

---

## Configuration Reference

### Available Configs

| Config | Description |
|--------|-------------|
| `demo_no_cache.yaml` | Simple pipeline, no embedding caching |
| `demo_onthefly_cache.yaml` | On-the-fly embedding caching |
| `demo_with_precompute.yaml` | Precomputed embeddings |
| `demo_loose_mode.yaml` | Loose matching (includes L3-only entities) |
| `default.yaml` | Full multi-database pipeline |

### Usage

```bash
# Default config (on-the-fly caching)
python demo.py

# Specific config
python demo.py --config configs/pipelines/demo_no_cache.yaml

# Custom entities file
python demo.py --entities my_entities.jsonl
```

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_sci_sm
```

### 2. Prepare Entities

Create `entities.jsonl`:
```jsonl
{"id": "MESH:D001943", "name": "TP53", "description": "Tumor protein p53", "entity_type": "gene", "popularity": 1000000, "aliases": ["p53", "tumor protein 53"]}
{"id": "MESH:D001943", "name": "BRCA1", "description": "Breast cancer gene 1", "entity_type": "gene", "popularity": 500000, "aliases": ["BRCA-1"]}
```

### 3. Run Pipeline

```python
import yaml
from src.core.factory import ProcessorFactory

# Load config
with open("configs/pipelines/demo_onthefly_cache.yaml") as f:
    config = yaml.safe_load(f)

# Create executor
executor = ProcessorFactory.create_from_dict(config)

# Load entities
executor.load_entities("entities.jsonl", target_layers=["dict"])

# Setup caching (optional)
executor.setup_l3_cache_writeback()

# Execute
result = executor.execute({"texts": ["TP53 mutations cause breast cancer"]})

# Process results
l0_output = result.get("l0_result")
for entity in l0_output.entities[0]:
    if entity.is_linked:
        print(f"{entity.mention_text} → {entity.linked_entity.label}")
        print(f"  Confidence: {entity.linked_entity.confidence:.2f}")
        print(f"  Entity ID: {entity.linked_entity.entity_id}")
```

### 4. Run Demo UI

```bash
python demo.py --config configs/pipelines/demo_onthefly_cache.yaml
```

---

## API Reference

### ProcessorFactory

```python
# Create from YAML file
executor = ProcessorFactory.create_from_file("config.yaml")

# Create from dict
executor = ProcessorFactory.create_from_dict(yaml_config)
```

### DAGExecutor

```python
# Execute pipeline
context = executor.execute({"texts": ["..."]})

# Load entities
executor.load_entities(
    filepath="entities.jsonl",
    target_layers=["dict", "redis"],  # Which layers to populate
    batch_size=1000,
    overwrite=False
)

# Precompute embeddings
stats = executor.precompute_embeddings(
    target_layers=["dict"],
    batch_size=32
)

# Setup L3 cache writeback
executor.setup_l3_cache_writeback()
```

### PipeContext

```python
# Get result by key
l0_result = context.get("l0_result")

# Save for debugging
context.to_json("debug_context.json")

# Restore
context = PipeContext.from_json("debug_context.json")
```

---

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{entitylinker2024,
  title={EntityLinker: A Modular DAG-based Entity Linking Framework},
  author={Knowledgator},
  year={2024},
  url={https://github.com/Knowledgator/EntityLinker}
}
```
