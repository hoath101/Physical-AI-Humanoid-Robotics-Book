"""
Ingestion pipeline for the Physical AI & Humanoid Robotics Book RAG Chatbot.
Handles the processing of book content into vector embeddings for retrieval.
"""

import os
import asyncio
import hashlib
import re
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import PyPDF2
import docx
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from config.ingestion_config import get_config_value
from services.ai_client import AIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookIngestionPipeline:
    """
    Comprehensive ingestion pipeline for processing book content into vector embeddings.
    Supports multiple file formats and chunking strategies.
    """

    def __init__(self, ai_client: AIClient, qdrant_client: QdrantClient = None):
        self.ai_client = ai_client

        # Get Qdrant configuration
        qdrant_url = get_config_value('QDRANT_URL', 'http://localhost:6333')
        qdrant_api_key = get_config_value('QDRANT_API_KEY', None)

        if qdrant_client is None:
            # Initialize Qdrant client with API key if provided and not using localhost
            # Avoid using API key with localhost to prevent "unsecure connection" warning
            if qdrant_api_key and not qdrant_url.startswith("http://localhost"):
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
            else:
                self.qdrant_client = QdrantClient(url=qdrant_url)
        else:
            self.qdrant_client = qdrant_client

        self.collection_name = get_config_value('QDRANT_COLLECTION_NAME', 'book_chunks')
        self.embedding_model = get_config_value('EMBEDDING_MODEL', 'text-embedding-3-small')
        self.max_chunk_size = int(get_config_value('MAX_CHUNK_SIZE', '1000'))
        self.overlap = int(get_config_value('CHUNK_OVERLAP', '100'))

    async def initialize_collection(self):
        """Initialize the Qdrant collection for storing book chunks."""
        try:
            # Check if collection exists - Qdrant client methods are synchronous
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            # Determine vector size based on AI provider
            vector_size = 1536  # Default for OpenAI
            if hasattr(self.ai_client, 'provider') and self.ai_client.provider == "gemini":
                vector_size = 768  # Gemini text-embedding-004

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
            logger.warning("Qdrant not available - ingestion will be skipped")
            # Re-raise the exception to maintain the original behavior
            # The calling function (run_ingestion_pipeline) will handle it appropriately
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

    def extract_text_from_md(self, md_path: str) -> str:
        """Extract text content from a Markdown file, removing headers and formatting."""
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Remove frontmatter if present (content between --- delimiters at the start)
                if content.startswith('---'):
                    frontmatter_end = content.find('---', 3)  # Find closing ---
                    if frontmatter_end != -1:
                        content = content[frontmatter_end + 3:].strip()

                # Remove markdown formatting while preserving the text content
                # Remove headers (# Header)
                content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
                # Remove bold and italic formatting
                content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
                content = re.sub(r'\*(.*?)\*', r'\1', content)
                content = re.sub(r'__(.*?)__', r'\1', content)
                content = re.sub(r'_(.*?)_', r'\1', content)
                # Remove inline code
                content = re.sub(r'`(.*?)`', r'\1', content)
                # Remove links [text](url) -> text
                content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
                # Remove images ![alt](url) -> alt
                content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', content)
                # Remove blockquotes
                content = re.sub(r'^>\s+', '', content, flags=re.MULTILINE)
                # Remove list markers
                content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)
                content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)

                # Clean up extra whitespace
                content = re.sub(r'\n\s*\n', '\n\n', content)  # Normalize multiple newlines
                content = content.strip()

                return content
        except Exception as e:
            logger.error(f"Error extracting text from Markdown {md_path}: {e}")
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
        text_len = len(text)

        while start < text_len:
            # Calculate end position (simple, no fancy boundary detection)
            end = min(start + max_chunk_size, text_len)

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
                        'total_length': text_len
                    }
                }
                chunks.append(chunk)
                chunk_id += 1

            # Move start position with overlap
            if overlap > 0 and overlap < end:
                start = end - overlap
            else:
                start = end

            # Safety: prevent infinite loop
            if end == text_len:
                break

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
        Create embeddings for a list of texts using AI client.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            total = len(texts)
            logger.info(f"Creating {total} embeddings in batch (parallel processing)...")

            # Use batch processing for much faster embedding creation
            embeddings = await self.ai_client.create_embeddings_batch(texts)

            logger.info(f"Successfully created {total} embeddings")
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
        logger.info(f"Step 1: Extracting text from file...")
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            content = self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            content = self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            content = self.extract_text_from_txt(file_path)
        elif file_ext == '.json':
            content = self.extract_text_from_json(file_path)
        elif file_ext == '.md':
            content = self.extract_text_from_md(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        logger.info(f"Step 2: Text extracted ({len(content)} chars). Splitting into chunks...")
        # Split content into chunks
        chunks = self.split_text(content, self.max_chunk_size, self.overlap)
        logger.info(f"Step 3: Created {len(chunks)} chunks. Adding metadata...")

        # Add metadata to each chunk
        for chunk in chunks:
            chunk['metadata'].update(self.extract_metadata_from_content(content, file_path))
            chunk['metadata']['source_file'] = str(file_path)

        logger.info(f"Step 4: Metadata added. Creating embeddings...")
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
        for idx, chunk in enumerate(chunks):
            # Generate unique ID from source file and chunk id
            unique_string = f"{chunk['metadata'].get('source_file', 'unknown')}_{chunk['metadata'].get('chunk_id', idx)}"
            chunk_id = hashlib.md5(unique_string.encode()).hexdigest()

            point = models.PointStruct(
                id=chunk_id,
                vector=chunk['embedding'],
                payload={
                    'id': chunk_id,  # Also store in payload for easier retrieval
                    'content': chunk['content'],
                    'metadata': chunk['metadata']
                }
            )
            points.append(point)

        # Upload points to Qdrant - synchronous call
        self.qdrant_client.upsert(
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
        supported_extensions = {'.pdf', '.docx', '.txt', '.json', '.md'}

        files_to_process = []
        for ext in supported_extensions:
            files_to_process.extend(directory.glob(f"*{ext}"))
            files_to_process.extend(directory.glob(f"**/*{ext}"))  # Include subdirectories

        total_chunks = 0
        processed_files = 0
        total_files = len(files_to_process)

        logger.info(f"Found {total_files} files to process")

        for idx, file_path in enumerate(files_to_process, 1):
            try:
                logger.info(f"Processing file {idx}/{total_files}: {file_path.name}")
                chunks = await self.process_file(str(file_path))
                if chunks:
                    logger.info(f"  -> Generated {len(chunks)} chunks, storing in Qdrant...")
                    await self.store_chunks_in_qdrant(chunks)
                    total_chunks += len(chunks)
                    logger.info(f"  -> Stored successfully!")
                    processed_files += 1
                    logger.info(f"Successfully processed {file_path} ({len(chunks)} chunks)")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue  # Continue with other files

        logger.info(f"Completed processing. Total: {processed_files} files, {total_chunks} chunks stored.")
        return processed_files


async def run_ingestion_pipeline(
    book_directory: str,
    ai_client: AIClient,
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
    pipeline = BookIngestionPipeline(ai_client)

    try:
        # Initialize Qdrant collection
        await pipeline.initialize_collection()

        # Process the directory
        processed_files = await pipeline.process_directory(book_directory)

        logger.info(f"Ingestion pipeline completed. Processed {processed_files} files.")
        return processed_files

    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {e}")
        # Check if this is a Qdrant connection error
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg or "timeout" in error_msg:
            logger.warning("Qdrant is not available. Ingestion skipped. Please start Qdrant server to enable document ingestion.")
            return 0  # Return 0 to indicate no files were processed due to Qdrant unavailability
        else:
            # Re-raise other types of errors
            raise


# Additional utility functions
async def reinitialize_collection(collection_name: Optional[str] = None):
    """Reinitialize the Qdrant collection (useful for clearing/restarting)."""
    pipeline = BookIngestionPipeline(AsyncOpenAI())  # Dummy client for initialization only
    if collection_name:
        pipeline.collection_name = collection_name
    await pipeline.initialize_collection()


def get_supported_file_types() -> List[str]:
    """Get list of supported file types for ingestion."""
    return ['.pdf', '.docx', '.txt', '.json', '.md']


if __name__ == "__main__":
    # Example usage (this would typically be called from the main API)
    import os
    from openai import AsyncOpenAI
    from datetime import datetime

    # This is just for testing - in production this would be called from main.py
    print("Ingestion pipeline module loaded. Use run_ingestion_pipeline() to process files.")