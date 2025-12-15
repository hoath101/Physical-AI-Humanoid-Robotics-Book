from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional
import logging

from .settings import settings

# Set up logging
logger = logging.getLogger(__name__)

class VectorDB:
    """
    Qdrant vector database client wrapper for managing vector collections and operations.
    """

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10.0
        )
        self.collection_name = "book_content_chunks"

    def initialize_collection(self):
        """
        Initialize the collection with appropriate vector configuration.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Size for OpenAI embeddings
                        distance=models.Distance.COSINE
                    )
                )

                # Create payload index for metadata filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="book_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chapter",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise

    def add_vectors(self, vectors, payloads, ids):
        """
        Add vectors with payloads to the collection.
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads
                )
            )
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            raise

    def search(self, vector, limit=5, filters=None):
        """
        Search for similar vectors with optional filters.
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=filters
            )
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

    def delete_by_book_id(self, book_id: str):
        """
        Delete all vectors associated with a specific book ID.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="book_id",
                                match=models.MatchValue(value=book_id)
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            logger.error(f"Error deleting vectors for book {book_id}: {e}")
            raise

# Create a singleton instance
vector_db = VectorDB()