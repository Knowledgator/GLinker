import json
import psycopg2
from psycopg2.extras import execute_batch

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="entities_db",
    user="postgres",
    password="postgres"
)
conn.autocommit = True
cursor = conn.cursor()

JSONL_FILE = 'entities.jsonl'

print("Creating tables...")
cursor.execute("""
    DROP TABLE IF EXISTS aliases CASCADE;
    DROP TABLE IF EXISTS entities CASCADE;
    
    CREATE TABLE entities (
        entity_id VARCHAR(255) PRIMARY KEY,
        label TEXT NOT NULL,
        entity_type VARCHAR(100),
        popularity INTEGER DEFAULT 0
    );
    
    CREATE TABLE aliases (
        id SERIAL PRIMARY KEY,
        entity_id VARCHAR(255) REFERENCES entities(entity_id),
        alias_text TEXT NOT NULL
    );
    
    CREATE INDEX idx_aliases_text ON aliases(LOWER(alias_text));
    CREATE INDEX idx_entities_popularity ON entities(popularity DESC);
    CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
""")
print("✓ Tables created!")

print("\nLoading data from JSONL...")
entities_data = []
aliases_data = []

with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        entity_id = data['id']
        label = data['name']
        entity_type = data['entity_type']
        popularity = data['popularity']
        
        entities_data.append((entity_id, label, entity_type, popularity))
        
        for alias in data['aliases']:
            aliases_data.append((entity_id, alias))

print(f"Inserting {len(entities_data)} entities...")
execute_batch(cursor, """
    INSERT INTO entities (entity_id, label, entity_type, popularity)
    VALUES (%s, %s, %s, %s)
""", entities_data)
print("✓ Entities inserted!")

print(f"Inserting {len(aliases_data)} aliases...")
execute_batch(cursor, """
    INSERT INTO aliases (entity_id, alias_text)
    VALUES (%s, %s)
""", aliases_data)
print("✓ Aliases inserted!")

cursor.execute("SELECT COUNT(*) FROM entities")
total_entities = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM aliases")
total_aliases = cursor.fetchone()[0]

cursor.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")
types_stats = cursor.fetchall()

print("\n" + "="*50)
print("DATABASE STATISTICS")
print("="*50)
print(f"Total entities: {total_entities}")
print(f"Total aliases: {total_aliases}")
print(f"\nBy type:")
for entity_type, count in types_stats:
    print(f"  - {entity_type}: {count}")
print("="*50)

cursor.close()
conn.close()
print("\n✓ Done!")