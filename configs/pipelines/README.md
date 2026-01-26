# Pipeline Configurations - Organized

This directory contains organized YAML configurations grouped by database architecture.

## Directory Structure

```
configs/pipelines/
├── postgres/                    # PostgreSQL only
│   ├── default.yaml            # Basic PostgreSQL setup
│   └── with_embeddings.yaml    # With precomputed embeddings
├── postgres_redis/              # PostgreSQL + Redis cache
│   ├── default.yaml            # Basic two-tier setup
│   └── with_embeddings.yaml    # With embedding cache
├── postgres_redis_elasticsearch/ # Three-tier setup
│   └── default.yaml            # Redis -> ES -> PostgreSQL
├── dict/                        # In-memory (for demo/dev)
│   ├── default.yaml            # Simple dict layer
│   ├── with_embeddings.yaml    # With on-the-fly embedding cache
│   ├── strict_mode.yaml        # Only L1-matched entities
│   └── loose_mode.yaml         # Includes L3 entities outside L1
└── legacy/                      # Old flat structure (deprecated)
    ├── default.yaml
    ├── demo_*.yaml
    └── ...
```

---

## PostgreSQL Configurations

### `postgres/default.yaml`
**Single-layer PostgreSQL setup**

- Storage: PostgreSQL only
- Caching: None
- Embeddings: No
- Use case: Production with persistent storage

```bash
python demo.py -c configs/pipelines/postgres/default.yaml
```

**Requirements:**
```bash
# Start PostgreSQL
cd scripts/database && docker-compose up -d postgres

# Load entities
python scripts/database/jsonl2postgresql.py
```

### `postgres/with_embeddings.yaml`
**PostgreSQL with precomputed embeddings**

- Storage: PostgreSQL
- Embeddings: Precomputed in PostgreSQL
- Model: BiEncoder (`gliner-deberta-base-v1-post`)
- Use case: High-performance BiEncoder inference

```python
from src.core.dag import DAGExecutor, DAGPipeline
import yaml

with open("configs/pipelines/postgres/with_embeddings.yaml") as f:
    pipeline = DAGPipeline(**yaml.safe_load(f))

executor = DAGExecutor(pipeline)

# Step 1: Load entities
executor.load_entities("entities.jsonl", target_layers=["postgres"])

# Step 2: Precompute embeddings
executor.precompute_embeddings(target_layers=["postgres"], batch_size=64)

# Step 3: Run demo
# python demo.py -c configs/pipelines/postgres/with_embeddings.yaml
```

---

## PostgreSQL + Redis Configurations

### `postgres_redis/default.yaml`
**Two-tier: Redis cache + PostgreSQL storage**

- Tier 1: Redis (priority 1, 1h TTL)
- Tier 2: PostgreSQL (priority 0, fallback)
- Embeddings: No
- Use case: Production with fast cache

```bash
# Start services
cd scripts/database && docker-compose up -d postgres redis

# Load entities
python scripts/database/jsonl2postgresql.py
python scripts/database/jsonl2redis.py

# Run
python demo.py -c configs/pipelines/postgres_redis/default.yaml
```

**Cache behavior:**
1. Query arrives → Check Redis (exact match)
2. Redis miss → Query PostgreSQL (exact + fuzzy)
3. PostgreSQL hit → Write back to Redis

### `postgres_redis/with_embeddings.yaml`
**Two-tier with embedding cache**

- Tier 1: Redis (caches entities + embeddings, 2h TTL)
- Tier 2: PostgreSQL (stores precomputed embeddings)
- Model: BiEncoder
- Use case: Maximum performance with persistent embeddings

```python
executor = DAGExecutor(pipeline)
executor.load_entities("entities.jsonl", target_layers=["postgres"])
executor.precompute_embeddings(target_layers=["postgres"])
# Redis will cache embeddings on first access
```

---

## Three-Tier Configuration

### `postgres_redis_elasticsearch/default.yaml`
**Redis → Elasticsearch → PostgreSQL**

- Tier 1: Redis (30 min TTL, exact match)
- Tier 2: Elasticsearch (24h TTL, full-text search)
- Tier 3: PostgreSQL (persistent storage)
- Use case: Production with fuzzy search

```bash
# Start all services
cd scripts/database && docker-compose up -d

# Load entities
python scripts/database/jsonl2postgresql.py
python scripts/database/jsonl2elasticsearch.py
python scripts/database/jsonl2redis.py

# Run
python demo.py -c configs/pipelines/postgres_redis_elasticsearch/default.yaml
```

**Cache hierarchy:**
1. Redis (fastest) → 2. Elasticsearch (fuzzy) → 3. PostgreSQL (fallback)

---

## In-Memory (Dict) Configurations

### `dict/default.yaml`
**Simple in-memory dictionary**

- Storage: In-memory dict
- Embeddings: No
- Use case: Demo, development, small datasets (<5000 entities)

```bash
python demo.py -c configs/pipelines/dict/default.yaml -e entities.jsonl
```

No database setup required! Entities loaded into memory.

### `dict/with_embeddings.yaml`
**In-memory with on-the-fly embedding cache**

- Storage: In-memory dict
- Embeddings: Computed on-the-fly, cached in L3
- Model: BiEncoder
- Use case: Demo with BiEncoder models

First run computes embeddings, subsequent runs use cache.

### `dict/strict_mode.yaml`
**Strict matching mode**

- L0 config: `strict_matching: true`
- Behavior: Only link entities that match L1 (spaCy NER) mentions
- Use case: High precision, trust spaCy NER

### `dict/loose_mode.yaml`
**Loose matching mode**

- L0 config: `strict_matching: false`
- Behavior: Include L3 entities even if L1 missed them
- Use case: Maximum recall, GLiNER finds additional entities

---

## Configuration Comparison

| Config | Storage | Cache | Embeddings | Matching | Use Case |
|--------|---------|-------|------------|----------|----------|
| `postgres/default` | PostgreSQL | None | No | Strict | Production, persistent |
| `postgres/with_embeddings` | PostgreSQL | None | Precomputed | Strict | BiEncoder, persistent |
| `postgres_redis/default` | PostgreSQL | Redis (1h) | No | Strict | Production, fast cache |
| `postgres_redis/with_embeddings` | PostgreSQL | Redis (2h) | Precomputed | Strict | BiEncoder, max perf |
| `postgres_redis_elasticsearch/default` | PostgreSQL | Redis + ES | No | Strict | Full-text search |
| `dict/default` | In-memory | None | No | Strict | Demo, dev |
| `dict/with_embeddings` | In-memory | L3 cache | On-the-fly | Strict | Demo BiEncoder |
| `dict/strict_mode` | In-memory | None | No | **Strict** | High precision |
| `dict/loose_mode` | In-memory | L3 cache | On-the-fly | **Loose** | Max recall |

---

## Quick Start Examples

### Development (No database)
```bash
python demo.py -c configs/pipelines/dict/default.yaml
```

### Production (PostgreSQL only)
```bash
cd scripts/database && docker-compose up -d postgres
python scripts/database/jsonl2postgresql.py
python demo.py -c configs/pipelines/postgres/default.yaml
```

### Production (PostgreSQL + Redis)
```bash
cd scripts/database && docker-compose up -d
python scripts/database/jsonl2postgresql.py
python demo.py -c configs/pipelines/postgres_redis/default.yaml
```

### BiEncoder with Embeddings
```bash
# Setup embeddings
python -c "
from src.core.dag import DAGExecutor, DAGPipeline
import yaml

with open('configs/pipelines/postgres/with_embeddings.yaml') as f:
    pipeline = DAGPipeline(**yaml.safe_load(f))

executor = DAGExecutor(pipeline)
executor.load_entities('entities.jsonl', target_layers=['postgres'])
executor.precompute_embeddings(target_layers=['postgres'])
"

# Run demo
python demo.py -c configs/pipelines/postgres/with_embeddings.yaml
```

---

## Migration from Legacy Configs

Old flat configs are in root `configs/pipelines/`:
- `default.yaml` → `postgres_redis_elasticsearch/default.yaml`
- `demo_no_cache.yaml` → `dict/default.yaml`
- `demo_onthefly_cache.yaml` → `dict/with_embeddings.yaml`
- `demo_with_precompute.yaml` → `postgres/with_embeddings.yaml`
- `demo_loose_mode.yaml` → `dict/loose_mode.yaml`

---

## Key Parameters

### L0 Matching Modes

**Strict mode** (`strict_matching: true`):
- Only entities at L1 mention positions
- Higher precision
- Misses entities that spaCy didn't detect

**Loose mode** (`strict_matching: false`):
- Includes L3 entities outside L1 mentions
- Higher recall
- May include false positives

### Cache Policies

- `always`: Write to cache on every search
- `miss`: Write only on cache miss
- `hit`: Write only on cache hit (refresh TTL)

### Search Modes

- `exact`: Exact string matching
- `fuzzy`: Fuzzy matching with similarity threshold
