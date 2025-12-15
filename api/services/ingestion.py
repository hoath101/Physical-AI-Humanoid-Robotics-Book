from typing import Dict, Any, List
import logging
from datetime import datetime
import hashlib

from api.models.document import BookContent
from api.utils.text_processing import chunk_text, extract_metadata_from_text, clean_text
from api.config.vector_db import vector_db
from api.config.settings import settings
from api.utils.validators import validate_ingestion_request
from api.models.request import IngestionRequest

# Set up logging
logger = logging.getLogger(__name__)

class IngestionService:
    """
    Service for ingesting and processing book content.
    """

    def __init__(self):
        self.vector_db = vector_db
        self.chunk_size = settings.chunk_size
        self.overlap_size = settings.overlap_size

    async def ingest_book_content(self, ingestion_request: IngestionRequest) -> Dict[str, Any]:
        """
        Process and ingest book content into the vector database.
        """
        start_time = datetime.now()

        # Validate the request
        validation_errors = validate_ingestion_request(
            ingestion_request.book_content,
            ingestion_request.book_metadata.dict(),
            ingestion_request.chunk_size,
            ingestion_request.overlap_size,
            ingestion_request.book_id
        )

        if validation_errors:
            raise ValueError(f"Validation errors: {', '.join(validation_errors)}")

        try:
            # Clean the book content
            cleaned_content = clean_text(ingestion_request.book_content)

            # Calculate content hash for change detection
            content_hash = hashlib.md5(cleaned_content.encode()).hexdigest()

            # Chunk the text
            chunks = chunk_text(
                cleaned_content,
                chunk_size=ingestion_request.chunk_size,
                overlap=ingestion_request.overlap_size
            )

            logger.info(f"Chunked book content into {len(chunks)} chunks")

            # Prepare data for vector database
            vectors = []
            payloads = []
            ids = []

            for idx, chunk in enumerate(chunks):
                # Create document ID
                doc_id = f"{ingestion_request.book_id}_chunk_{idx}"

                # Extract basic metadata
                chunk_metadata = extract_metadata_from_text(chunk)

                # Create payload with book metadata
                payload = {
                    "id": doc_id,
                    "book_id": ingestion_request.book_id,
                    "text": chunk,
                    "title": ingestion_request.book_metadata.title,
                    "author": ingestion_request.book_metadata.author,
                    "isbn": ingestion_request.book_metadata.isbn,
                    "edition": ingestion_request.book_metadata.edition,
                    "chunk_index": idx,
                    "content_hash": content_hash,
                    "created_at": datetime.now().isoformat(),
                    "chapter": "",  # Will be populated if found in content
                    "page": None,   # Will be populated if found in content
                    "section": "",  # Will be populated if found in content
                    "paragraph_id": f"p{idx}",  # Simple paragraph ID based on chunk index
                    **chunk_metadata  # Add any extracted metadata
                }

                # Add chapter, section if available in the original content
                # This is a simple heuristic - in a real implementation, you'd want more sophisticated parsing
                if "potential_headers" in chunk_metadata and chunk_metadata["potential_headers"]:
                    # Use the last header as the most relevant one for this chunk
                    last_header = chunk_metadata["potential_headers"][-1][1]
                    payload["section"] = last_header

                # Prepare vectors for Qdrant
            chunk_texts = [payload["text"] for payload in payloads]

            # Generate embeddings for each chunk
            from api.services.embedding import embedding_service
            embeddings = await embedding_service.generate_embeddings(chunk_texts)

            # Prepare vectors list
            vectors = []
            for i, embedding in enumerate(embeddings):
                vectors.append(embedding)

            # Initialize the collection if needed
            self.vector_db.initialize_collection()

            # Store the content with embeddings in the vector database
            self.vector_db.add_vectors(vectors, payloads, ids)

            logger.info(f"Stored {len(ids)} chunks with embeddings in vector database")

            # Calculate processing time
            processing_time = datetime.now() - start_time

            result = {
                "status": "success",
                "message": "Book content successfully ingested and indexed",
                "book_id": ingestion_request.book_id,
                "chunks_processed": len(chunks),
                "processing_time": f"{processing_time.total_seconds():.2f}s",
                "content_hash": content_hash
            }

            logger.info(f"Ingestion completed for book {ingestion_request.book_id}")
            return result

        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            raise

    async def delete_book_content(self, book_id: str) -> Dict[str, Any]:
        """
        Delete all content associated with a specific book ID.
        """
        try:
            self.vector_db.delete_by_book_id(book_id)

            result = {
                "status": "success",
                "message": f"Content for book {book_id} deleted successfully",
                "book_id": book_id
            }

            logger.info(f"Deletion completed for book {book_id}")
            return result

        except Exception as e:
            logger.error(f"Error during deletion: {str(e)}")
            raise

    async def update_book_content(self, ingestion_request: IngestionRequest) -> Dict[str, Any]:
        """
        Update existing book content by deleting old content and ingesting new content.
        """
        # First, delete existing content for this book
        try:
            await self.delete_book_content(ingestion_request.book_id)
        except Exception:
            # If deletion fails, it might be because the book doesn't exist yet
            # That's okay, we'll just proceed with ingestion
            logger.info(f"Book {ingestion_request.book_id} doesn't exist yet, proceeding with fresh ingestion")

        # Then, ingest the new content
        result = await self.ingest_book_content(ingestion_request)

        result["message"] = f"Book {ingestion_request.book_id} updated successfully"
        return result

# Create a singleton instance
ingestion_service = IngestionService()