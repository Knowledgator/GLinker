import json
import sys
import redis

JSONL_FILE = 'entities.jsonl'

# Try to connect
print("Checking Redis connection...")
try:
    client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True,
        socket_connect_timeout=5
    )
    client.ping()
    print("✓ Redis is running!")
except redis.ConnectionError:
    print("❌ ERROR: Redis is not running!")
    print("\nTo start Redis, run:")
    print("  docker run -d --name redis -p 6379:6379 redis:7-alpine")
    print("\nOr start your local Redis service:")
    print("  sudo systemctl start redis")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: Cannot connect to Redis: {e}")
    sys.exit(1)

# Clear existing data
print("\nClearing existing data...")
client.flushdb()
print("✓ Database cleared!")

# Load data from JSONL
print("\nLoading data from JSONL...")
entities_count = 0
aliases_count = 0

with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        entity_id = data['id']
        
        # Store entity data as hash
        client.hset(f"entity:{entity_id}", mapping={
            'entity_id': entity_id,
            'label': data['name'],
            'entity_type': data['entity_type'],
            'popularity': data['popularity']
        })
        entities_count += 1
        
        # Store aliases
        for alias in data['aliases']:
            alias_lower = alias.lower()
            
            # alias:xxx -> set of entity IDs
            client.sadd(f"alias:{alias_lower}", entity_id)
            
            # entity:xxx:aliases -> set of aliases
            client.sadd(f"entity:{entity_id}:aliases", alias)
            aliases_count += 1

print(f"✓ Inserted {entities_count} entities")
print(f"✓ Inserted {aliases_count} aliases")

# Stats
entity_keys = list(client.scan_iter(match="entity:*", count=1000))
# Filter out alias sets
entity_hashes = [k for k in entity_keys if not k.endswith(':aliases')]
total_entities = len(entity_hashes)

alias_keys = list(client.scan_iter(match="alias:*", count=1000))
total_aliases = len(alias_keys)

# Count by type
types_count = {}
for key in entity_hashes:
    entity_data = client.hgetall(key)
    entity_type = entity_data.get('entity_type', 'unknown')
    types_count[entity_type] = types_count.get(entity_type, 0) + 1

print("\n" + "="*50)
print("REDIS DATABASE READY")
print("="*50)
print(f"Total entities: {total_entities}")
print(f"Total unique aliases: {total_aliases}")
print(f"\nBy type:")
for entity_type, count in types_count.items():
    print(f"  - {entity_type}: {count}")
print("="*50)

# Show some examples
print("\nExample data:")
for key in list(client.scan_iter(match="entity:*", count=3)):
    if not key.endswith(':aliases'):
        data = client.hgetall(key)
        print(f"  {data['label']} ({data['entity_type']})")

client.close()
print("\n✓ Redis setup complete!")