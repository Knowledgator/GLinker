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

### 2. Load Data
```bash
# Install Python dependencies
pip install psycopg2-binary elasticsearch redis

# Load data into each database
python scripts/database/setup_postgres.py
python scripts/database/setup_elasticsearch.py
python scripts/database/setup_redis.py
```

### 3. Test
```bash
python scripts/test_l2_processor.py
```

## Using the Automated Script
```bash
# Start all services
bash scripts/database/setup_all.sh

# Load data
python scripts/database/setup_postgres.py
python scripts/database/setup_elasticsearch.py
python scripts/database/setup_redis.py
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