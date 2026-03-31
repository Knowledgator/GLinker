#!/bin/bash

set -e

echo "=========================================="
echo "CLEANING UP EXISTING CONTAINERS"
echo "=========================================="

# Stop and remove all database containers
echo "Stopping and removing existing containers..."
docker stop postgres elasticsearch redis 2>/dev/null || true
docker rm -f postgres elasticsearch redis 2>/dev/null || true

echo "✓ Cleanup complete!"
echo ""

echo "=========================================="
echo "STARTING ALL DATABASE SERVICES"
echo "=========================================="

# Check Docker - improved check
if ! docker --version &> /dev/null; then
    echo "❌ Docker not found! Please install Docker first."
    echo ""
    echo "Install Docker with:"
    echo "  sudo apt update"
    echo "  sudo apt install docker.io"
    echo "  sudo usermod -aG docker $USER"
    echo ""
    echo "Or use snap (not recommended):"
    echo "  sudo snap install docker"
    echo ""
    echo "Then log out and log back in."
    exit 1
fi

echo "✓ Docker is installed: $(docker --version)"

# Check Docker permissions
if ! docker ps > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Docker permission issue detected!"
    echo "Adding current user to docker group..."
    sudo usermod -aG docker $USER
    echo ""
    echo "✓ User added to docker group"
    echo ""
    echo "⚠️  Changes will take effect after you:"
    echo "  1. Log out and log back in"
    echo "  OR"
    echo "  2. Run: newgrp docker"
    echo "  Then run this script again: ./setup_all.sh"
    exit 0
fi

# Start PostgreSQL
echo ""
echo "Starting PostgreSQL container..."
docker run -d \
  --name postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=entities_db \
  postgres:15-alpine

echo "Waiting for PostgreSQL to be ready..."
sleep 5
for i in {1..20}; do
  if docker exec postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✓ PostgreSQL is ready on localhost:5432"
    break
  fi
  sleep 1
done

# Start Elasticsearch
echo ""
echo "Starting Elasticsearch container..."
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.11.0

echo "Waiting for Elasticsearch to be ready (takes ~30 seconds)..."
for i in {1..60}; do
  if curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo "✓ Elasticsearch is ready on http://localhost:9200"
    break
  fi
  sleep 2
done

# Start Redis
echo ""
echo "Starting Redis container..."
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine

sleep 3
echo "✓ Redis is ready on localhost:6379"

# Done
echo ""
echo "=========================================="
echo "✓ ALL SERVICES ARE RUNNING!"
echo "=========================================="
echo ""
echo "Services:"
echo "  🐘 PostgreSQL:     localhost:5432"
echo "     User: postgres, Password: postgres, DB: entities_db"
echo ""
echo "  🔍 Elasticsearch:  http://localhost:9200"
echo ""
echo "  🔴 Redis:          localhost:6379"
echo ""
echo "=========================================="
echo ""
echo "Container status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Next steps - Load entities using Python API:"
echo ""
echo "  from glinker.core.factory import ProcessorFactory"
echo "  import yaml"
echo ""
echo "  with open('configs/pipelines/dict/strict_mode.yaml') as f:"
echo "      config = yaml.safe_load(f)"
echo ""
echo "  executor = ProcessorFactory.create_from_dict(config)"
echo "  executor.load_entities('data/pubmesh_ontology.jsonl', target_layers=['dict'])"
echo ""
echo "See docs/DATABASE_SETUP.md for detailed instructions"
echo ""
echo "To stop services:"
echo "  docker stop postgres elasticsearch redis"
echo ""
echo "To remove services:"
echo "  docker rm -f postgres elasticsearch redis"
echo ""