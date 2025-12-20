"""
Quick script to check Qdrant collection status
"""
from qdrant_client import QdrantClient
from config.ingestion_config import get_config_value

# Get configuration
qdrant_url = get_config_value('QDRANT_URL', 'http://localhost:6333')
qdrant_api_key = get_config_value('QDRANT_API_KEY', None)
collection_name = get_config_value('QDRANT_COLLECTION_NAME', 'book_chunks')

# Initialize Qdrant client
if qdrant_api_key and not qdrant_url.startswith("http://localhost"):
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )
else:
    qdrant_client = QdrantClient(url=qdrant_url)

print("="*60)
print("Qdrant Collection Status")
print("="*60)

try:
    # Get collection info
    collection_info = qdrant_client.get_collection(collection_name)

    print(f"\nCollection Name: {collection_name}")
    print(f"Vector Size: {collection_info.config.params.vectors.size}")
    print(f"Distance Metric: {collection_info.config.params.vectors.distance}")
    print(f"Total Points: {collection_info.points_count}")
    print(f"Indexed Vectors: {collection_info.indexed_vectors_count}")

    if collection_info.points_count == 0:
        print("\n[WARNING] Collection is EMPTY! No documents have been ingested.")
        print("          Run: python ingest_docs_simple.py")
    else:
        print(f"\n[SUCCESS] Collection has {collection_info.points_count} documents!")

except Exception as e:
    print(f"\n[ERROR] Could not access collection: {e}")

print("="*60)
