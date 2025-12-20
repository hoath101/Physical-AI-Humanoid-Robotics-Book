"""
Script to reset the Qdrant collection for Gemini embeddings.
This deletes the existing collection and creates a new one with the correct vector dimensions.
"""
import asyncio
from qdrant_client import QdrantClient
from config.ingestion_config import get_config_value
from services.ai_client import AIClient

async def reset_collection():
    """Delete and recreate the Qdrant collection with correct dimensions"""

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

    # Initialize AI client to check provider
    ai_client = AIClient()

    print(f"[DELETE] Deleting existing collection: {collection_name}")

    try:
        # Delete existing collection
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"[SUCCESS] Deleted collection: {collection_name}")
    except Exception as e:
        print(f"[WARNING] Collection may not exist or error deleting: {e}")

    # Determine vector size based on provider
    if ai_client.provider == "gemini":
        vector_size = 768  # Gemini text-embedding-004
        print(f"[INFO] Using Gemini vector size: {vector_size}")
    else:
        vector_size = 1536  # OpenAI text-embedding-3-small
        print(f"[INFO] Using OpenAI vector size: {vector_size}")

    # Create new collection
    from qdrant_client.http import models

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

    print(f"[SUCCESS] Created new collection: {collection_name} with vector size: {vector_size}")
    print(f"\n[COMPLETE] Collection reset complete! You can now run your server.")
    print(f"   Note: You'll need to re-ingest your documents if you had any.")

if __name__ == "__main__":
    asyncio.run(reset_collection())
