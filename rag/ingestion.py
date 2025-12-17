"""
Ingestion pipeline for the Physical AI & Humanoid Robotics Book RAG Chatbot.
Handles the processing of book content into vector embeddings for retrieval.
"""

import os
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import PyPDF2
import docx
import json
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from config.ingestion_config import get_config_value

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookIngestionPipeline:
    """
    Comprehensive ingestion pipeline for processing book content into vector embeddings.
    Supports multiple file formats and chunking strategies.
    """

    def __init__(self, openai_client: OpenAI, qdrant_client: QdrantClient = None):
        self.openai_client = openai_client
        self.qdrant_client = qdrant_client or QdrantClient(
            url=get_config_value('QDRANT_URL', 'http://localhost:6333')
        )
        self.collection_name = get_config_value('QDRANT_COLLECTION_NAME', 'book_chunks')
        self.embedding_model = get_config_value('EMBEDDING_MODEL', 'text-embedding-3-small')
        self.max_chunk_size = int(get_config_value('MAX_CHUNK_SIZE', '1000'))
        self.overlap = int(get_config_value('CHUNK_OVERLAP', '100'))

    async def initialize_collection(self):
        """Initialize the Qdrant collection for storing book chunks."""
        try:
            # Check if collection exists
            collections = await self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Default size for text-embedding-3-small
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
        return text

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text content from a DOCX file."""
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {docx_path}: {e}")
            raise

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text content from a TXT file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {txt_path}: {e}")
            raise

    def extract_text_from_json(self, json_path: str) -> str:
        """Extract text content from a JSON file (assuming book content format)."""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Handle different JSON structures
                if isinstance(data, str):
                    return data
                elif isinstance(data, list):
                    return " ".join([str(item) for item in data])
                elif isinstance(data, dict):
                    # Look for common book content keys
                    content_keys = ['content', 'text', 'body', 'chapters', 'sections']
                    for key in content_keys:
                        if key in data:
                            if isinstance(data[key], str):
                                return data[key]
                            elif isinstance(data[key], list):
                                return " ".join([str(item) for item in data[key]])
                    # If no common keys found, convert entire dict to string
                    return json.dumps(data)
                else:
                    return str(data)
        except Exception as e:
            logger.error(f"Error extracting text from JSON {json_path}: {e}")
            raise

    def split_text(self, text: str, max_chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: The text to split
            max_chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # Determine the end position
            end = start + max_chunk_size

            # If we're near the end, include the rest
            if end >= len(text):
                end = len(text)
            else:
                # Try to break at sentence boundary
                while end > start + max_chunk_size // 2 and end < len(text) and text[end] not in '.!?':
                    end += 1
                # If we couldn't find a good break point, just use max_chunk_size
                if end <= start + max_chunk_size // 2:
                    end = start + max_chunk_size

            # Extract the chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = {
                    'id': f'chunk_{chunk_id}',
                    'content': chunk_text,
                    'metadata': {
                        'start_pos': start,
                        'end_pos': end,
                        'chunk_id': chunk_id,
                        'total_length': len(text)
                    }
                }
                chunks.append(chunk)
                chunk_id += 1

            # Move start position with overlap
            start = end - overlap if overlap < end else end

        return chunks

    def extract_metadata_from_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from content and file path.

        Args:
            content: The text content
            file_path: Path to the source file

        Returns:
            Dictionary of metadata
        """
        # Basic metadata extraction
        metadata = {
            'file_path': str(file_path),
            'file_name': Path(file_path).name,
            'file_extension': Path(file_path).suffix,
            'file_size': os.path.getsize(file_path),
            'content_length': len(content),
            'created_at': str(datetime.now()),
        }

        # Try to extract more specific book-related metadata
        try:
            # Look for chapter/section patterns in the content
            import re

            # Look for chapter titles (common patterns)
            chapter_patterns = [
                r'Chapter\s+(\d+)[\s:\-\—]*([^\n\r]+)',
                r'#\s*Chapter\s+(\d+)[\s:\-\—]*([^\n\r]+)',
                r'##\s*Chapter\s+(\d+)[\s:\-\—]*([^\n\r]+)',
                r'CHAPTER\s+(\d+)[\s:\-\—]*([^\n\r]+)',
            ]

            for pattern in chapter_patterns:
                match = re.search(pattern, content[:1000], re.IGNORECASE)  # Look in first 1000 chars
                if match:
                    metadata['chapter_number'] = match.group(1)
                    metadata['chapter_title'] = match.group(2).strip()
                    break

            # Look for section titles
            section_patterns = [
                r'Section\s+(\d+)[\s:\-\—]*([^\n\r]+)',
                r'#\s*([^\n\r]+)',
                r'##\s*([^\n\r]+)',
            ]

            for pattern in section_patterns:
                match = re.search(pattern, content[:500], re.IGNORECASE)  # Look in first 500 chars
                if match and 'section_title' not in metadata:
                    metadata['section_title'] = match.group(1).strip()
                    break

        except Exception as e:
            logger.warning(f"Error extracting content metadata: {e}")

        return metadata

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    async def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single file into chunks with embeddings.

        Args:
            file_path: Path to the file to process

        Returns:
            List of processed chunks with embeddings
        """
        logger.info(f"Processing file: {file_path}")

        # Extract text based on file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            content = self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            content = self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            content = self.extract_text_from_txt(file_path)
        elif file_ext == '.json':
            content = self.extract_text_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Split content into chunks
        chunks = self.split_text(content, self.max_chunk_size, self.overlap)

        # Add metadata to each chunk
        for chunk in chunks:
            chunk['metadata'].update(self.extract_metadata_from_content(content, file_path))
            chunk['metadata']['source_file'] = str(file_path)

        # Create embeddings for all chunks
        if chunks:
            texts = [chunk['content'] for chunk in chunks]
            embeddings = await self.create_embeddings(texts)

            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i]

        logger.info(f"Processed {len(chunks)} chunks from {file_path}")
        return chunks

    async def store_chunks_in_qdrant(self, chunks: List[Dict[str, Any]]):
        """
        Store processed chunks in Qdrant vector database.

        Args:
            chunks: List of processed chunks to store
        """
        if not chunks:
            return

        points = []
        for chunk in chunks:
            point = models.PointStruct(
                id=hashlib.md5(f"{chunk['metadata']['source_file']}_{chunk['metadata']['chunk_id']}".encode()).hexdigest(),
                vector=chunk['embedding'],
                payload={
                    'content': chunk['content'],
                    'metadata': chunk['metadata']
                }
            )
            points.append(point)

        # Upload points to Qdrant
        await self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Stored {len(points)} chunks in Qdrant collection {self.collection_name}")

    async def process_directory(self, directory_path: str) -> int:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to directory containing book files

        Returns:
            Number of files processed
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")

        # Supported file extensions
        supported_extensions = {'.pdf', '.docx', '.txt', '.json'}

        files_to_process = []
        for ext in supported_extensions:
            files_to_process.extend(directory.glob(f"*{ext}"))
            files_to_process.extend(directory.glob(f"**/*{ext}"))  # Include subdirectories

        total_chunks = 0
        processed_files = 0

        for file_path in files_to_process:
            try:
                chunks = await self.process_file(str(file_path))
                if chunks:
                    await self.store_chunks_in_qdrant(chunks)
                    total_chunks += len(chunks)
                    processed_files += 1
                    logger.info(f"Successfully processed {file_path} ({len(chunks)} chunks)")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue  # Continue with other files

        logger.info(f"Completed processing. Total: {processed_files} files, {total_chunks} chunks stored.")
        return processed_files


async def run_ingestion_pipeline(
    book_directory: str,
    openai_client: OpenAI,
    max_chunk_size: int = 1000,
    overlap: int = 100
):
    """
    Main function to run the book ingestion pipeline.

    Args:
        book_directory: Directory containing book files to process
        openai_client: Initialized OpenAI client
        max_chunk_size: Maximum size of text chunks
        overlap: Overlap between chunks
    """
    logger.info(f"Starting ingestion pipeline for directory: {book_directory}")

    # Update config values for this run
    import config.ingestion_config as config_module
    config_module._config_values['MAX_CHUNK_SIZE'] = str(max_chunk_size)
    config_module._config_values['CHUNK_OVERLAP'] = str(overlap)

    # Create ingestion pipeline instance
    pipeline = BookIngestionPipeline(openai_client)

    try:
        # Initialize Qdrant collection
        await pipeline.initialize_collection()

        # Process the directory
        processed_files = await pipeline.process_directory(book_directory)

        logger.info(f"Ingestion pipeline completed. Processed {processed_files} files.")
        return processed_files

    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {e}")
        raise


# Additional utility functions
async def reinitialize_collection(collection_name: Optional[str] = None):
    """Reinitialize the Qdrant collection (useful for clearing/restarting)."""
    pipeline = BookIngestionPipeline(OpenAI())  # Dummy client for initialization only
    if collection_name:
        pipeline.collection_name = collection_name
    await pipeline.initialize_collection()


def get_supported_file_types() -> List[str]:
    """Get list of supported file types for ingestion."""
    return ['.pdf', '.docx', '.txt', '.json']


if __name__ == "__main__":
    # Example usage (this would typically be called from the main API)
    import os
    from openai import OpenAI
    from datetime import datetime

    # This is just for testing - in production this would be called from main.py
    print("Ingestion pipeline module loaded. Use run_ingestion_pipeline() to process files.")