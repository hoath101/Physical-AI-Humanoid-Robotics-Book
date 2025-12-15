import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, openai_client: OpenAI):
        # Initialize Qdrant client for vector search
        self.qdrant_client = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )

        # Set up OpenAI client for embedding generation
        self.openai_client = openai_client

        # Collection name for book chunks
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_chunks")

    async def initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            collections = await self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size for OpenAI embeddings
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
                )

                # Create index for faster filtering on section field
                await self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="section",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                logger.info(f"Created collection '{self.collection_name}' in Qdrant with indexes")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists in Qdrant")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def retrieve_relevant_chunks(
        self,
        query: str,
        book_section: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant text chunks from the vector database based on the query."""
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_embedding(query)

            # Prepare filters
            filters = []
            if book_section:
                filters.append(models.FieldCondition(
                    key="section",
                    match=models.MatchValue(value=book_section)
                ))

            # Convert filters to a single filter if there are any
            filter_obj = None
            if filters:
                filter_obj = models.Filter(must=filters)

            # Perform semantic search in Qdrant
            search_results = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_obj,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            retrieved_chunks = []
            for hit in search_results:
                chunk_data = {
                    "content": hit.payload.get("content", ""),
                    "metadata": {
                        "id": hit.id,
                        "section": hit.payload.get("section", ""),
                        "chapter": hit.payload.get("chapter", ""),
                        "page": hit.payload.get("page", ""),
                        "source_file": hit.payload.get("source_file", ""),
                        "score": hit.score
                    }
                }
                retrieved_chunks.append(chunk_data)

            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks for query: {query[:50]}...")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            raise

    async def add_document_chunk(
        self,
        content: str,
        doc_id: str,
        section: str,
        chapter: str = "",
        page: str = "",
        source_file: str = ""
    ):
        """Add a document chunk to the vector database."""
        try:
            # Generate embedding for the content
            embedding = await self.generate_embedding(content)

            # Prepare the payload
            payload = {
                "content": content,
                "section": section,
                "chapter": chapter,
                "page": page,
                "source_file": source_file,
                "doc_id": doc_id
            }

            # Upsert the point in Qdrant
            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            logger.info(f"Added document chunk to Qdrant with ID: {doc_id}")

        except Exception as e:
            logger.error(f"Error adding document chunk: {str(e)}")
            raise

    async def bulk_add_document_chunks(self, chunks_data: List[Dict[str, Any]]) -> int:
        """
        Bulk add multiple document chunks to the vector database.

        Args:
            chunks_data: List of dictionaries containing content, doc_id, section, etc.

        Returns:
            Number of successfully added chunks
        """
        try:
            # Prepare points for bulk insertion
            points = []
            for chunk_data in chunks_data:
                content = chunk_data['content']
                doc_id = chunk_data['doc_id']
                section = chunk_data.get('section', '')
                chapter = chunk_data.get('chapter', '')
                page = chunk_data.get('page', '')
                source_file = chunk_data.get('source_file', '')

                # Generate embedding for the content
                embedding = await self.generate_embedding(content)

                # Prepare the payload
                payload = {
                    "content": content,
                    "section": section,
                    "chapter": chapter,
                    "page": page,
                    "source_file": source_file,
                    "doc_id": doc_id
                }

                # Create point struct
                point = models.PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)

            # Bulk upsert points in batches (Qdrant recommends keeping batches under 64KB)
            batch_size = 50  # Adjust based on average chunk size
            successful_count = 0

            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                successful_count += len(batch)

                logger.info(f"Bulk added batch of {len(batch)} chunks ({successful_count}/{len(points)} total)")

            logger.info(f"Successfully added {successful_count} document chunks to Qdrant")
            return successful_count

        except Exception as e:
            logger.error(f"Error in bulk adding document chunks: {str(e)}")
            raise

    async def delete_document_chunk(self, doc_id: str):
        """Delete a document chunk from the vector database."""
        try:
            await self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[doc_id]
                )
            )
            logger.info(f"Deleted document chunk from Qdrant with ID: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document chunk: {str(e)}")
            raise

    async def search_by_content(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks by content similarity."""
        try:
            query_embedding = await self.generate_embedding(text)

            search_results = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for hit in search_results:
                results.append({
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload,
                    "score": hit.score
                })

            return results
        except Exception as e:
            logger.error(f"Error searching by content: {str(e)}")
            raise

    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document."""
        try:
            records = await self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False
            )

            if records:
                return records[0].payload
            return None
        except Exception as e:
            logger.error(f"Error retrieving document metadata: {str(e)}")
            raise

    async def get_all_sections(self) -> List[str]:
        """Get all unique sections in the collection."""
        try:
            # Get all points and extract unique sections
            scroll_result = await self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on expected number of documents
                with_payload=True,
                with_vectors=False
            )

            sections = set()
            for record in scroll_result[0]:  # scroll returns (records, next_page_offset)
                section = record.payload.get("section", "")
                if section:
                    sections.add(section)

            return list(sections)
        except Exception as e:
            logger.error(f"Error retrieving sections: {str(e)}")
            raise

    async def get_chunk_count(self) -> int:
        """Get the total number of chunks in the collection."""
        try:
            collection_info = await self.qdrant_client.get_collection(
                collection_name=self.collection_name
            )
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting chunk count: {str(e)}")
            raise

    async def clear_collection(self):
        """Clear all points from the collection (use with caution!)."""
        try:
            await self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[]
                    )
                )
            )
            logger.info(f"Cleared collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise