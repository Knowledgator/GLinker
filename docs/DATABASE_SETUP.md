# Complete Database Setup Guide

## Table of Contents
1. [Docker Installation](#docker-installation)
2. [Database Setup](#database-setup)
3. [Entity Data Loading](#entity-data-loading)
4. [Verification](#verification)
5. [Container Management](#container-management)
6. [Troubleshooting](#troubleshooting)

---

## Docker Installation

### Check if Docker is Already Installed

```bash
docker --version
docker ps
```

If both commands work, skip to [Database Setup](#database-setup).

### Install Docker on Ubuntu/Debian

#### Method 1: Using apt (Recommended)

```bash
# Update package index
sudo apt update

# Install dependencies
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
sudo docker --version
```

#### Method 2: Using docker.io package

```bash
sudo apt update
sudo apt install -y docker.io docker-compose

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Install Docker on Fedora/RHEL/CentOS

```bash
# Install Docker
sudo dnf install -y docker docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Install Docker on macOS

```bash
# Using Homebrew
brew install --cask docker

# Or download Docker Desktop from:
# https://www.docker.com/products/docker-desktop
```

### Configure Docker Permissions

Add your user to the docker group to run Docker without sudo:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply changes (choose one):
# Option 1: Log out and log back in
# Option 2: Run new shell with docker group
newgrp docker

# Option 3: Reboot system
sudo reboot
```

Verify permissions:
```bash
docker ps  # Should work without sudo
```

### Install Docker Compose (if not included)

```bash
# Check if docker-compose is installed
docker-compose --version

# If not installed:
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

---

## Database Setup

### Method 1: Docker Compose (Recommended)

**Advantages:**
- Single command to start all services
- Persistent volumes (data survives container restarts)
- Health checks
- Automatic restart on failure
- Network isolation

**Steps:**

1. Navigate to database scripts directory:
```bash
cd scripts/database
```

2. Start all services:
```bash
docker-compose up -d
```

3. Check service status:
```bash
docker-compose ps
```

Expected output:
```
NAME                 IMAGE                    STATUS          PORTS
el_postgres          postgres:15-alpine       Up (healthy)    0.0.0.0:5432->5432/tcp
el_elasticsearch     elasticsearch:8.11.0     Up (healthy)    0.0.0.0:9200->9200/tcp, 0.0.0.0:9300->9300/tcp
el_redis             redis:7-alpine           Up (healthy)    0.0.0.0:6379->6379/tcp
```

4. View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f postgres
docker-compose logs -f elasticsearch
docker-compose logs -f redis
```

5. Wait for services to be ready:
```bash
# PostgreSQL
docker-compose exec postgres pg_isready -U postgres

# Elasticsearch
curl -s http://localhost:9200/_cluster/health | grep status

# Redis
docker-compose exec redis redis-cli ping
```

### Method 2: Automated Bash Script

```bash
cd scripts/database
chmod +x setup_all.sh
./setup_all.sh
```

The script will:
- Check Docker installation
- Configure permissions if needed
- Stop and remove old containers
- Start PostgreSQL, Elasticsearch, Redis
- Wait for services to be ready
- Display connection info

### Method 3: Manual Container Setup

#### Prerequisites

System requirements for Elasticsearch:
```bash
# Set vm.max_map_count (required for Elasticsearch)
sudo sysctl -w vm.max_map_count=262144

# Make permanent (add to /etc/sysctl.conf)
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
```

#### PostgreSQL

```bash
# Create volume for persistent storage
docker volume create postgres_data

# Run container
docker run -d \
  --name postgres \
  --restart unless-stopped \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=entities_db \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15-alpine

# Wait for PostgreSQL to be ready
sleep 5
docker exec postgres pg_isready -U postgres

# Expected output: /var/run/postgresql:5432 - accepting connections
```

#### Elasticsearch

```bash
# Create volume
docker volume create elasticsearch_data

# Run container
docker run -d \
  --name elasticsearch \
  --restart unless-stopped \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  -e "bootstrap.memory_lock=true" \
  --ulimit memlock=-1:-1 \
  -v elasticsearch_data:/usr/share/elasticsearch/data \
  elasticsearch:8.11.0

# Wait for Elasticsearch (takes ~30-60 seconds)
echo "Waiting for Elasticsearch..."
for i in {1..60}; do
  if curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo "Elasticsearch is ready!"
    break
  fi
  echo -n "."
  sleep 2
done

# Check cluster health
curl -s http://localhost:9200/_cluster/health?pretty
```

#### Redis

```bash
# Create volume
docker volume create redis_data

# Run container
docker run -d \
  --name redis \
  --restart unless-stopped \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:7-alpine redis-server --appendonly yes

# Test connection
sleep 3
docker exec redis redis-cli ping
# Expected output: PONG
```

#### Verify All Services

```bash
# Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test connections
docker exec postgres pg_isready -U postgres
curl -s http://localhost:9200/_cluster/health | grep -o '"status":"[^"]*"'
docker exec redis redis-cli ping
```

---

## Entity Data Loading

### Prepare Entity Data

Ensure you have `entities.jsonl` in the project root:

```bash
# Check file exists
ls -lh entities.jsonl

# View first entity
head -n 1 entities.jsonl | jq .
```

Expected format:
```json
{
  "entity_id": "Q123",
  "label": "BRCA1",
  "description": "Breast cancer type 1 susceptibility protein",
  "entity_type": "gene",
  "popularity": 1000000,
  "aliases": ["BRCA-1", "breast cancer 1"]
}
```

### Method 1: Python Scripts (Simple)

Load entities into specific databases:

```bash
# PostgreSQL
python scripts/database/jsonl2postgresql.py

# Elasticsearch
python scripts/database/jsonl2elasticsearch.py

# Redis
python scripts/database/jsonl2redis.py
```

### Method 2: DAG Executor (Flexible)

```python
from src.core.dag import DAGExecutor, DAGPipeline
import yaml

# Load pipeline configuration
with open("configs/pipelines/postgres_redis_basic.yaml") as f:
    config = yaml.safe_load(f)
    pipeline = DAGPipeline(**config)

# Create executor
executor = DAGExecutor(pipeline, verbose=True)

# Load entities into PostgreSQL and Redis
executor.load_entities(
    filepath="entities.jsonl",
    target_layers=["postgres", "redis"],
    batch_size=1000,
    overwrite=True  # Drop existing data
)

print("Entities loaded successfully!")
```

### Method 3: Load with Precomputed Embeddings

For BiEncoder models, precompute embeddings for faster inference:

```python
from src.core.dag import DAGExecutor, DAGPipeline
import yaml

# Use embedding-enabled config
with open("configs/pipelines/postgres_redis_embeddings.yaml") as f:
    pipeline = DAGPipeline(**yaml.safe_load(f))

executor = DAGExecutor(pipeline, verbose=True)

# Step 1: Load entities into PostgreSQL
print("Loading entities...")
executor.load_entities(
    filepath="entities.jsonl",
    target_layers=["postgres"],
    batch_size=1000
)

# Step 2: Precompute embeddings (this takes time!)
print("Precomputing embeddings...")
executor.precompute_embeddings(
    target_layers=["postgres"],
    batch_size=64  # Adjust based on GPU memory
)

print("Setup complete! Embeddings cached in PostgreSQL.")
```

---

## Verification

### PostgreSQL

```bash
# Connect to database
docker exec -it postgres psql -U postgres -d entities_db

# Inside psql:
\dt                              # List tables
SELECT COUNT(*) FROM entities;   # Count entities
SELECT * FROM entities LIMIT 5;  # View sample data
\q                               # Exit
```

One-liner:
```bash
docker exec -it postgres psql -U postgres -d entities_db -c "SELECT COUNT(*) FROM entities;"
```

Check embeddings (if precomputed):
```bash
docker exec -it postgres psql -U postgres -d entities_db -c "
  SELECT entity_id, label,
         CASE WHEN embedding IS NOT NULL THEN 'Yes' ELSE 'No' END as has_embedding,
         embedding_model_id
  FROM entities
  LIMIT 5;
"
```

### Elasticsearch

```bash
# Check cluster health
curl -s http://localhost:9200/_cluster/health?pretty

# Count entities
curl -s http://localhost:9200/entities/_count | jq .

# Search entities
curl -s -X GET "http://localhost:9200/entities/_search?q=BRCA1&pretty"

# View index mapping
curl -s http://localhost:9200/entities/_mapping?pretty
```

### Redis

```bash
# Interactive CLI
docker exec -it redis redis-cli

# Inside redis-cli:
DBSIZE                           # Count keys
KEYS entity:*                    # List entity keys (use cautiously in production)
GET entity:BRCA1                 # Get specific entity
TTL entity:BRCA1                 # Check TTL
INFO memory                      # Memory usage
exit

# One-liner examples
docker exec redis redis-cli DBSIZE
docker exec redis redis-cli GET "entity:BRCA1"
docker exec redis redis-cli INFO memory | grep used_memory_human
```

### Test Pipeline End-to-End

```python
from src.core.dag import DAGExecutor, DAGPipeline
from src.l1.models import L1Input
import yaml

# Load config
with open("configs/pipelines/postgres_redis_basic.yaml") as f:
    pipeline = DAGPipeline(**yaml.safe_load(f))

# Create executor
executor = DAGExecutor(pipeline, verbose=True)

# Test query
test_text = "BRCA1 mutations are associated with breast cancer risk."
result = executor.execute(L1Input(texts=[test_text]))

# Check results
l0_result = result.get("l0_result")
print(f"Found {len(l0_result.entities)} entities")
for entity in l0_result.entities:
    print(f"  - {entity.mention}: {entity.linked_entity.label if entity.linked_entity else 'Not linked'}")
```

---

## Container Management

### Docker Compose Commands

```bash
# Start services
docker-compose up -d

# Stop services (containers remain)
docker-compose stop

# Start stopped services
docker-compose start

# Restart services
docker-compose restart

# Stop and remove containers
docker-compose down

# Stop, remove containers AND delete volumes (ALL DATA LOST!)
docker-compose down -v

# View logs
docker-compose logs -f              # All services
docker-compose logs -f postgres     # Single service
docker-compose logs --tail=100      # Last 100 lines

# Execute commands
docker-compose exec postgres psql -U postgres -d entities_db
docker-compose exec redis redis-cli
```

### Manual Container Management

```bash
# List containers
docker ps                  # Running containers
docker ps -a               # All containers

# Start/stop
docker start postgres elasticsearch redis
docker stop postgres elasticsearch redis
docker restart postgres

# Remove containers
docker rm -f postgres elasticsearch redis

# Remove containers + volumes (data loss!)
docker volume rm postgres_data elasticsearch_data redis_data
```

### Cleanup

```bash
# Remove all stopped containers
docker container prune -f

# Remove unused volumes
docker volume prune -f

# Remove unused images
docker image prune -a -f

# Complete cleanup (ALL DOCKER DATA!)
docker system prune -a --volumes -f
```

---

## Troubleshooting

### Docker Not Found

```bash
# Check if Docker is installed
which docker

# If not found, install Docker (see installation section above)
```

### Permission Denied

```bash
# Error: permission denied while trying to connect to the Docker daemon socket
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended)
sudo docker ps
```

### Port Already in Use

```bash
# Find process using port
sudo lsof -i :5432   # PostgreSQL
sudo lsof -i :9200   # Elasticsearch
sudo lsof -i :6379   # Redis

# Kill process (if safe)
sudo kill -9 <PID>

# Or stop old containers
docker stop $(docker ps -q --filter ancestor=postgres:15-alpine)
docker stop $(docker ps -q --filter ancestor=elasticsearch:8.11.0)
docker stop $(docker ps -q --filter ancestor=redis:7-alpine)
```

### Elasticsearch Won't Start

**Issue: vm.max_map_count too low**

```bash
# Check current value
sysctl vm.max_map_count

# Set temporarily
sudo sysctl -w vm.max_map_count=262144

# Set permanently
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Issue: Not enough memory**

```bash
# Check container logs
docker logs elasticsearch

# Reduce memory allocation in docker-compose.yml or:
docker run ... -e "ES_JAVA_OPTS=-Xms256m -Xmx256m" ...
```

### PostgreSQL Connection Refused

```bash
# Check if PostgreSQL is ready
docker exec postgres pg_isready -U postgres

# Check logs
docker logs postgres

# Try connecting
docker exec -it postgres psql -U postgres -d entities_db

# If database doesn't exist
docker exec postgres createdb -U postgres entities_db
```

### Redis Connection Issues

```bash
# Test connection
docker exec redis redis-cli ping

# Check if Redis is running
docker ps | grep redis

# View logs
docker logs redis

# Restart Redis
docker restart redis
```

### Out of Disk Space

```bash
# Check Docker disk usage
docker system df

# Remove unused data
docker system prune -a --volumes

# Check volume sizes
docker volume ls
docker volume inspect postgres_data | grep Mountpoint
```

### Container Exits Immediately

```bash
# Check exit code and error
docker logs <container_name>

# Inspect container
docker inspect <container_name>

# Run with interactive mode for debugging
docker run -it postgres:15-alpine /bin/sh
```

### Network Issues

```bash
# Check Docker networks
docker network ls

# Inspect network
docker network inspect bridge

# Test connectivity between containers
docker exec postgres ping elasticsearch
```

### Python Connection Errors

```python
# PostgreSQL
import psycopg2
try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="entities_db",
        user="postgres",
        password="postgres"
    )
    print("PostgreSQL: Connected!")
except Exception as e:
    print(f"PostgreSQL Error: {e}")

# Elasticsearch
from elasticsearch import Elasticsearch
try:
    es = Elasticsearch(["http://localhost:9200"])
    print(f"Elasticsearch: {es.ping()}")
except Exception as e:
    print(f"Elasticsearch Error: {e}")

# Redis
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    print(f"Redis: {r.ping()}")
except Exception as e:
    print(f"Redis Error: {e}")
```

### Reset Everything

Complete reset (WARNING: deletes all data):

```bash
# Stop and remove all EntityLinker containers
docker-compose down -v

# Remove volumes
docker volume rm postgres_data elasticsearch_data redis_data

# Remove containers manually (if not using docker-compose)
docker rm -f postgres elasticsearch redis

# Start fresh
cd scripts/database
docker-compose up -d

# Reload data
python scripts/database/jsonl2postgresql.py
```

---

## Quick Reference

### Connection Details

| Service | Host | Port | Credentials |
|---------|------|------|-------------|
| PostgreSQL | localhost | 5432 | user: postgres, password: postgres, db: entities_db |
| Elasticsearch | localhost | 9200 | No auth |
| Redis | localhost | 6379 | No password |

### Useful Commands

```bash
# Health check all services
docker exec postgres pg_isready -U postgres && \
curl -s http://localhost:9200/_cluster/health | grep -o '"status":"[^"]*"' && \
docker exec redis redis-cli ping

# View all logs
docker logs postgres --tail 50
docker logs elasticsearch --tail 50
docker logs redis --tail 50

# Backup PostgreSQL
docker exec postgres pg_dump -U postgres entities_db > backup.sql

# Restore PostgreSQL
docker exec -i postgres psql -U postgres entities_db < backup.sql
```
