import sys
sys.path.append('.')

from config import POSTGRES_CONFIG, ELASTICSEARCH_CONFIG, REDIS_CONFIG
from src.database.postgres import PostgresDatabase
from src.database.elasticsearch import ElasticsearchDatabase
from src.database.redis import RedisDatabase

def main():
    print("=" * 60)
    print("STEP 1: INITIALIZING DATABASE SCHEMAS")
    print("=" * 60)
    
    print("\n1. Connecting to PostgreSQL...")
    postgres = PostgresDatabase(POSTGRES_CONFIG)
    postgres.connect()
    print("✓ Connected")
    
    print("\n2. Creating PostgreSQL schema...")
    postgres.initialize_schema()
    print("✓ Schema created")
    
    print("\n3. Connecting to Elasticsearch...")
    es = ElasticsearchDatabase(ELASTICSEARCH_CONFIG)
    es.connect()
    print("✓ Connected")
    
    print("\n4. Creating Elasticsearch index...")
    es.initialize_schema()
    print("✓ Index created")
    
    print("\n5. Connecting to Redis...")
    redis = RedisDatabase(REDIS_CONFIG)
    redis.connect()
    print("✓ Connected")
    
    postgres.close()
    es.close()
    redis.close()
    
    print("\n" + "=" * 60)
    print("✓ ALL DATABASES INITIALIZED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()