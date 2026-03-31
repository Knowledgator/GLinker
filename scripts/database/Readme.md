# Database Setup Guide

Quick guide to set up PostgreSQL, Elasticsearch, and Redis for the Entity Linker.

## Prerequisites

- Docker installed
- Python 3.8+
- `entities.jsonl` file in project root

## Quick Start

### 1. Start Database Services
```bash
# Stop any existing local PostgreSQL
sudo systemctl stop postgresql

# Start all databases via Docker
docker run -d \
  --name postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=entities_db \
  postgres:15-alpine

docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.11.0

docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Check they're running
docker ps
```

### 2. Load Data Using Python API

```python
from glinker.core.factory import ProcessorFactory
import yaml

# Load your pipeline configuration
with open("configs/pipelines/dict/strict_mode.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Create executor
executor = ProcessorFactory.create_from_dict(config)

# Load entities from JSONL file
# Automatically detects and loads into configured layers (dict, postgres, redis, elasticsearch)
executor.load_entities(
    "data/pubmesh_ontology.jsonl",
    target_layers=["dict"],  # or ["postgres", "redis"], etc.
    batch_size=1000,
    overwrite=True  # Set to False to append instead of replacing
)

print("✅ Entities loaded successfully!")
```

### 3. Verify Data Loaded

```python
# Check entity count in each layer
l2_processor = executor.processors["l2"]
counts = l2_processor.component.count_entities()
print(f"Entity counts: {counts}")

# Test a query
result = executor.execute({"texts": ["BRCA1 is a gene."]})
print(result)
```

## Connection Details

- **PostgreSQL**: `localhost:5432`
  - User: `postgres`
  - Password: `postgres`
  - Database: `entities_db`

- **Elasticsearch**: `http://localhost:9200`

- **Redis**: `localhost:6379`

## Managing Services
```bash
# Stop all
docker stop postgres elasticsearch redis

# Start all
docker start postgres elasticsearch redis

# Remove all
docker rm -f postgres elasticsearch redis

# View logs
docker logs postgres
docker logs elasticsearch
docker logs redis
```

## Troubleshooting

**Port already in use:**
```bash
# Stop local services
sudo systemctl stop postgresql
sudo systemctl stop elasticsearch
sudo systemctl stop redis

# Or change ports in docker run commands
```

**Docker permission denied:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Wait for Elasticsearch to be ready:**
```bash
# Elasticsearch takes ~30 seconds to start
curl http://localhost:9200
```