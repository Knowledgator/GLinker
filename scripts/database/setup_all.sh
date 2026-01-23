#!/bin/bash

set -e

echo "=========================================="
echo "STARTING ALL DATABASE SERVICES"
echo "=========================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found! Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker $USER
    echo "✓ Docker installed!"
    echo ""
    echo "⚠️  Please log out and log back in, then run this script again."
    exit 0
fi

echo "✓ Docker is installed"

# Check Docker permissions
if ! docker ps > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Docker permission issue detected!"
    echo "Adding current user to docker group..."
    sudo usermod -aG docker $USER
    echo ""
    echo "✓ User added to docker group"
    echo ""
    echo "To apply changes, run one of:"
    echo "  1. Log out and log back in"
    echo "  2. Run: newgrp docker && ./setup_all.sh"
    echo "  3. Reboot your system"
    exit 0
fi

# Start PostgreSQL
echo ""
echo "Starting PostgreSQL container..."
docker rm -f postgres 2>/dev/null || true
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
docker rm -f elasticsearch 2>/dev/null || true
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
docker rm -f redis 2>/dev/null || true
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
echo "Next steps - Load data into databases:"
echo ""
echo "  python scripts/database/setup_postgres.py"
echo "  python scripts/database/setup_elasticsearch.py"
echo "  python scripts/database/setup_redis.py"
echo ""
echo "To stop services:"
echo "  docker stop postgres elasticsearch redis"
echo ""