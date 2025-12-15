import asyncio
import hashlib
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import aiofiles
from openai import OpenAI
from qdrant_client.http import models
from db.database import db_manager, DocumentMetadata

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BookChunk:
    """Data class representing a book chunk."""
    id: str
    content: str
    document_id: str
    section: str
    chapter: str
    page_numbers: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]

class BookIngestionPipeline:
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client

        # Import here to avoid circular dependencies
        from rag.retriever import RAGRetriever
        self.rag_retriever = RAGRetriever(openai_client)

    async def initialize_system(self):
        """Initialize the ingestion system components."""
        try:
            # Initialize Qdrant collection
            await self.rag_retriever.initialize_collection()

            # Connect to database
            await db_manager.connect()

            logger.info("Book ingestion system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ingestion system: {str(e)}")
            raise

    def _generate_chunk_id(self, content: str, document_id: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk based on content hash."""
        content_hash = hashlib.sha256(f"{content}{document_id}{chunk_index}".encode()).hexdigest()
        return f"chunk_{content_hash[:16]}"

    def _generate_document_id(self, title: str, source_file: str) -> str:
        """Generate a unique ID for a document."""
        content_hash = hashlib.sha256(f"{title}{source_file}".encode()).hexdigest()
        return f"doc_{content_hash[:16]}"

    def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 100,
        section: str = "",
        chapter: str = "",
        page_numbers: str = "",
        source_file: str = ""
    ) -> List[BookChunk]:
        """
        Split text into chunks of specified size with overlap.

        Args:
            text: The text to chunk
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            section: Section identifier
            chapter: Chapter identifier
            page_numbers: Page numbers
            source_file: Source file path

        Returns:
            List of BookChunk objects
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + max_chunk_size

            # If we're near the end, include the rest
            if end > len(text):
                end = len(text)

            # Extract the chunk
            chunk_content = text[start:end]

            # Create chunk ID
            document_id = self._generate_document_id(section, source_file)
            chunk_id = self._generate_chunk_id(chunk_content, document_id, chunk_index)

            # Create BookChunk object
            chunk = BookChunk(
                id=chunk_id,
                content=chunk_content,
                document_id=document_id,
                section=section,
                chapter=chapter,
                page_numbers=page_numbers,
                source_file=source_file,
                chunk_index=chunk_index,
                metadata={
                    "section": section,
                    "chapter": chapter,
                    "page_numbers": page_numbers,
                    "source_file": source_file,
                    "chunk_index": chunk_index
                }
            )

            chunks.append(chunk)

            # Move start position forward, accounting for overlap
            start = end - overlap
            chunk_index += 1

            # Prevent infinite loop if overlap is too large
            if overlap >= max_chunk_size:
                break

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    async def process_markdown_file(
        self,
        file_path: str,
        section: str,
        chapter: str = "",
        max_chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[BookChunk]:
        """
        Process a markdown file and split it into chunks.

        Args:
            file_path: Path to the markdown file
            section: Section identifier
            chapter: Chapter identifier
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of BookChunk objects
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Extract page numbers if available in the file content
            page_numbers = self._extract_page_numbers(content)

            # Chunk the content
            chunks = self.chunk_text(
                content,
                max_chunk_size=max_chunk_size,
                overlap=overlap,
                section=section,
                chapter=chapter,
                page_numbers=page_numbers,
                source_file=str(file_path)
            )

            logger.info(f"Processed markdown file {file_path}: {len(chunks)} chunks generated")
            return chunks
        except Exception as e:
            logger.error(f"Error processing markdown file {file_path}: {str(e)}")
            raise

    def _extract_page_numbers(self, content: str) -> str:
        """Extract page numbers from content if available."""
        # This is a simple implementation - could be enhanced based on actual content format
        # Look for common page number patterns in the content
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.strip().startswith('Page ') or line.strip().endswith('Page'):
                return line.strip()
        return ""

    async def process_docusaurus_docs(
        self,
        docs_dir: str,
        max_chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[BookChunk]:
        """
        Process all markdown files in a Docusaurus docs directory.

        Args:
            docs_dir: Path to the Docusaurus docs directory
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of all BookChunk objects from all files
        """
        all_chunks = []

        docs_path = Path(docs_dir)
        markdown_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))

        logger.info(f"Found {len(markdown_files)} markdown files in {docs_dir}")

        for md_file in markdown_files:
            # Determine section and chapter from file path
            relative_path = md_file.relative_to(docs_path)
            parts = str(relative_path).split(os.sep)

            section = parts[0] if len(parts) > 0 else "unknown"
            chapter = parts[1] if len(parts) > 1 else "unknown"

            # Skip certain files that are not content (like sidebar configs)
            if any(skip in str(md_file) for skip in ['sidebar', 'toc']):
                continue

            try:
                file_chunks = await self.process_markdown_file(
                    str(md_file),
                    section=section,
                    chapter=chapter,
                    max_chunk_size=max_chunk_size,
                    overlap=overlap
                )
                all_chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Error processing file {md_file}: {str(e)}")
                continue

        logger.info(f"Processed {len(all_chunks)} total chunks from {docs_dir}")
        return all_chunks

    async def ingest_chunks(self, chunks: List[BookChunk]):
        """
        Ingest chunks into Qdrant and Neon Postgres.

        Args:
            chunks: List of BookChunk objects to ingest
        """
        logger.info(f"Starting ingestion of {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            try:
                # Add chunk to Qdrant (vector database)
                await self.rag_retriever.add_document_chunk(
                    content=chunk.content,
                    doc_id=chunk.id,
                    section=chunk.section,
                    chapter=chunk.chapter,
                    page=chunk.page_numbers,
                    source_file=chunk.source_file
                )

                # Save chunk metadata to Neon Postgres
                await db_manager.save_document_chunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    section=chunk.section,
                    chapter=chunk.chapter,
                    page_numbers=chunk.page_numbers,
                    source_file=chunk.source_file,
                    embedding_id=chunk.id  # Using chunk ID as embedding ID
                )

                # Update document metadata in database
                document_metadata = DocumentMetadata(
                    id=chunk.document_id,
                    title=chunk.section,
                    section=chunk.section,
                    chapter=chunk.chapter,
                    page_numbers=chunk.page_numbers,
                    source_file=chunk.source_file,
                    created_at="",
                    updated_at="",
                    embedding_count=0  # This will be updated in the DB manager
                )
                await db_manager.save_document_metadata(document_metadata)

                if (i + 1) % 50 == 0:  # Log progress every 50 chunks
                    logger.info(f"Ingested {i + 1}/{len(chunks)} chunks...")

            except Exception as e:
                logger.error(f"Error ingesting chunk {chunk.id}: {str(e)}")
                continue

        logger.info(f"Completed ingestion of {len(chunks)} chunks")

    async def ingest_book_from_directory(
        self,
        book_directory: str,
        max_chunk_size: int = 1000,
        overlap: int = 100
    ):
        """
        Complete ingestion pipeline for a book directory.

        Args:
            book_directory: Path to the book directory containing markdown files
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        logger.info(f"Starting book ingestion from directory: {book_directory}")

        # Initialize the system
        await self.initialize_system()

        # Process all markdown files in the directory
        chunks = await self.process_docusaurus_docs(
            book_directory,
            max_chunk_size=max_chunk_size,
            overlap=overlap
        )

        # Ingest all chunks
        await self.ingest_chunks(chunks)

        logger.info("Book ingestion completed successfully")

    async def update_document_metadata(self, document_id: str, title: str, section: str):
        """Update document metadata in the database."""
        metadata = DocumentMetadata(
            id=document_id,
            title=title,
            section=section,
            chapter="",
            page_numbers="",
            source_file="",
            created_at="",
            updated_at="",
            embedding_count=0
        )
        await db_manager.save_document_metadata(metadata)

# Convenience function to run the ingestion pipeline
async def run_ingestion_pipeline(
    book_directory: str,
    openai_client: OpenAI,
    max_chunk_size: int = 1000,
    overlap: int = 100
):
    """
    Run the complete book ingestion pipeline.

    Args:
        book_directory: Path to the book directory containing markdown files
        openai_client: Initialized OpenAI client
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    """
    ingestion_pipeline = BookIngestionPipeline(openai_client)
    await ingestion_pipeline.ingest_book_from_directory(
        book_directory,
        max_chunk_size=max_chunk_size,
        overlap=overlap
    )