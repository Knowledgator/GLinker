import json
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import ConnectionError

JSONL_FILE = 'entities.jsonl'

# Try to connect
print("Checking Elasticsearch connection...")
try:
    es = Elasticsearch(["http://localhost:9200"], request_timeout=5)
    es.info()  # Test connection
    print("✓ Elasticsearch is running!")
except ConnectionError:
    print("❌ ERROR: Elasticsearch is not running!")
    print("\nTo start Elasticsearch, run:")
    print("  docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e 'discovery.type=single-node' -e 'xpack.security.enabled=false' elasticsearch:8.11.0")
    print("\nOr start your local Elasticsearch service:")
    print("  sudo systemctl start elasticsearch")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: Cannot connect to Elasticsearch: {e}")
    sys.exit(1)

index_name = "entities"

# Delete if exists
if es.indices.exists(index=index_name):
    print(f"Deleting existing index '{index_name}'...")
    es.indices.delete(index=index_name)
    print("✓ Index deleted!")

# Create index
print(f"\nCreating index '{index_name}'...")
mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "entity_id": {"type": "keyword"},
            "label": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}}
            },
            "entity_type": {"type": "keyword"},
            "popularity": {"type": "integer"},
            "description": {"type": "text"},
            "aliases": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}}
            }
        }
    }
}

es.indices.create(index=index_name, body=mapping)
print("✓ Index created!")

# Load data
print("\nLoading data from JSONL...")
documents = []

with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        documents.append({
            "_index": index_name,
            "_id": data['id'],
            "_source": {
                "entity_id": data['id'],
                "label": data['name'],
                "entity_type": data['entity_type'],
                "popularity": data['popularity'],
                "description": data.get('description', ''),
                "aliases": data['aliases']
            }
        })

print(f"Inserting {len(documents)} documents...")
success, failed = bulk(es, documents, raise_on_error=False)
print(f"✓ Inserted: {success}, Failed: {failed}")

es.indices.refresh(index=index_name)

# Stats
stats = es.count(index=index_name)
total_docs = stats['count']

agg_result = es.search(
    index=index_name,
    body={
        "size": 0,
        "aggs": {
            "types": {
                "terms": {"field": "entity_type", "size": 20}
            }
        }
    }
)

print("\n" + "="*50)
print("ELASTICSEARCH INDEX READY")
print("="*50)
print(f"Total documents: {total_docs}")
print(f"\nBy type:")
for bucket in agg_result['aggregations']['types']['buckets']:
    print(f"  - {bucket['key']}: {bucket['doc_count']}")
print("="*50)

print("\n✓ Elasticsearch setup complete!")