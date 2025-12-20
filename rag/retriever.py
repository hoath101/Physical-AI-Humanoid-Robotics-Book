"""
RAG (Retrieval-Augmented Generation) retriever for the Physical AI & Humanoid Robotics Book RAG Chatbot.
Handles vector search and document retrieval for question answering.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from config.ingestion_config import get_config_value
from services.ai_client import AIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    RAG (Retrieval-Augmented Generation) retriever that performs vector similarity search
    to find relevant book content for answering user questions.
    """

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client

        # Get Qdrant configuration
        qdrant_url = get_config_value('QDRANT_URL', 'http://localhost:6333')
        qdrant_api_key = get_config_value('QDRANT_API_KEY', None)

        # Initialize Qdrant client with API key if provided and not using localhost
        # Avoid using API key with localhost to prevent "unsecure connection" warning
        if qdrant_api_key and not qdrant_url.startswith("http://localhost"):
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            self.qdrant_client = QdrantClient(url=qdrant_url)

        self.collection_name = get_config_value('QDRANT_COLLECTION_NAME', 'book_chunks')
        self.embedding_model = get_config_value('EMBEDDING_MODEL', 'text-embedding-3-small')
        self.default_top_k = int(get_config_value('DEFAULT_TOP_K', '5'))
        self.search_threshold = float(get_config_value('SEARCH_THRESHOLD', '0.3'))

    async def initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists - Qdrant client methods are synchronous, not async
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            # Determine vector size based on AI provider
            vector_size = 1536  # Default for OpenAI text-embedding-3-small
            if self.ai_client.provider == "gemini":
                vector_size = 768  # Gemini text-embedding-004 uses 768 dimensions

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name} with vector size: {vector_size}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            # Log the error but don't raise it - allow the app to start even if Qdrant is not available
            # This allows the application to start for development purposes
            logger.warning("Qdrant not available - search functionality will be disabled until Qdrant is running")

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        try:
            embedding = await self.ai_client.create_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    async def retrieve_relevant_chunks(
        self,
        query: str,
        book_section: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant text chunks for a given query.

        Args:
            query: User's question or query text
            book_section: Optional section/chapter to limit search to
            top_k: Number of top results to return (defaults to config value)
            threshold: Minimum similarity threshold for results

        Returns:
            List of relevant chunks with content and metadata
        """
        if not query.strip():
            return []

        top_k = top_k or self.default_top_k
        threshold = threshold or self.search_threshold

        try:
            # Create embedding for the query
            query_embedding = await self.create_embedding(query)

            # Build search filter if book_section is specified
            search_filter = None
            if book_section:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.section_title",
                            match=MatchValue(value=book_section)
                        )
                    ]
                )

            # Perform vector search in Qdrant - synchronous call
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=threshold
            )

            # Process results
            relevant_chunks = []
            for result in search_results:
                if result.score >= threshold:
                    chunk = {
                        'id': result.payload.get('id', str(result.id)),  # Use payload ID or convert result.id to string
                        'content': result.payload.get('content', ''),
                        'metadata': result.payload.get('metadata', {}),
                        'score': result.score
                    }
                    relevant_chunks.append(chunk)

            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: {query[:50]}...")
            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            # Return empty list if Qdrant is not available, rather than raising an exception
            logger.warning("Qdrant not available - returning empty results for search")
            return []

    async def get_chunk_count(self) -> int:
        """
        Get the total number of chunks in the collection.

        Returns:
            Total number of chunks stored
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0

    async def get_all_sections(self) -> List[str]:
        """
        Get all unique sections/chapters from the stored metadata.

        Returns:
            List of unique section titles
        """
        try:
            # This would require a scroll operation to get all unique sections
            # For now, return an empty list or implement proper aggregation
            # In a real implementation, you might want to maintain a separate index of sections
            return []  # Placeholder - implement based on your specific metadata structure
        except Exception as e:
            logger.error(f"Error getting all sections: {e}")
            return []

    async def retrieve_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks based on metadata filters.

        Args:
            metadata_filters: Dictionary of metadata field-value pairs to filter by
            top_k: Number of results to return

        Returns:
            List of matching chunks
        """
        top_k = top_k or self.default_top_k

        try:
            # Build filter from metadata
            conditions = []
            for key, value in metadata_filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )

            search_filter = Filter(must=conditions) if conditions else None

            # Perform search with filter - synchronous call
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_filter=search_filter,
                limit=top_k
            )

            # Process results
            chunks = []
            for result in search_results:
                chunk = {
                    'id': result.id,
                    'content': result.payload.get('content', ''),
                    'metadata': result.payload.get('metadata', {}),
                    'score': result.score
                }
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error retrieving by metadata: {e}")
            raise

    async def retrieve_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve specific chunks by their IDs.

        Args:
            ids: List of chunk IDs to retrieve

        Returns:
            List of chunks with matching IDs
        """
        try:
            # Retrieve points by ID - synchronous call
            points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False
            )

            # Process results
            chunks = []
            for point in points:
                chunk = {
                    'id': point.id,
                    'content': point.payload.get('content', ''),
                    'metadata': point.payload.get('metadata', {}),
                }
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error retrieving by IDs: {e}")
            raise

    async def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional filters.

        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            List of semantically similar chunks
        """
        top_k = top_k or self.default_top_k
        threshold = threshold or self.search_threshold

        try:
            # Create embedding for the query
            query_embedding = await self.create_embedding(query)

            # Build filter if provided
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                search_filter = Filter(must=conditions) if conditions else None

            # Perform semantic search - synchronous call
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=threshold
            )

            # Process results
            chunks = []
            for result in search_results:
                if result.score >= threshold:
                    chunk = {
                        'id': result.id,
                        'content': result.payload.get('content', ''),
                        'metadata': result.payload.get('metadata', {}),
                        'score': result.score
                    }
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'collection_name': self.collection_name,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_type': collection_info.config.params.vectors.distance,
                'total_points': collection_info.points_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'full_scan_threshold': collection_info.config.params.optimizers_config.full_scan_threshold
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    async def delete_chunks_by_metadata(self, metadata_filters: Dict[str, Any]) -> int:
        """
        Delete chunks that match metadata filters.

        Args:
            metadata_filters: Dictionary of metadata field-value pairs to match for deletion

        Returns:
            Number of deleted chunks
        """
        try:
            # First, find the IDs of points to delete
            conditions = []
            for key, value in metadata_filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )

            search_filter = Filter(must=conditions) if conditions else None

            # Get points that match the filter - synchronous call
            points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=10000  # Limit to avoid memory issues
            )

            # Extract IDs
            ids_to_delete = [point.id for point in points[0]]

            if ids_to_delete:
                # Delete the points - synchronous call
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=ids_to_delete
                    )
                )

            logger.info(f"Deleted {len(ids_to_delete)} chunks matching filters")
            return len(ids_to_delete)

        except Exception as e:
            logger.error(f"Error deleting chunks by metadata: {e}")
            raise


# Additional utility functions
async def create_retriever(ai_client: AIClient) -> RAGRetriever:
    """Create and initialize a RAG retriever instance."""
    retriever = RAGRetriever(ai_client)
    await retriever.initialize_collection()
    return retriever


async def get_relevant_chunks(
    query: str,
    ai_client: AIClient,
    book_section: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Get relevant chunks for a query using a temporary retriever."""
    retriever = await create_retriever(ai_client)
    return await retriever.retrieve_relevant_chunks(query, book_section, top_k)


if __name__ == "__main__":
    # Example usage (this would typically be called from the main API)
    print("RAG Retriever module loaded. Use RAGRetriever class for vector search operations.")