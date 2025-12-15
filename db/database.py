import os
import asyncpg
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import logging
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Data class for document metadata."""
    id: str
    title: str
    section: str
    chapter: str
    page_numbers: str
    source_file: str
    created_at: str
    updated_at: str
    embedding_count: int

class DatabaseManager:
    def __init__(self):
        self.connection_pool = None
        self.db_url = os.getenv("NEON_DB_URL")

    async def connect(self):
        """Establish connection to the Neon Postgres database."""
        try:
            if not self.db_url:
                raise ValueError("NEON_DB_URL environment variable is not set")

            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(
                dsn=self.db_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )

            logger.info("Successfully connected to Neon Postgres database")

            # Initialize tables if they don't exist
            await self._initialize_tables()

        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    async def _initialize_tables(self):
        """Initialize required tables in the database."""
        async with self.connection_pool.acquire() as conn:
            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    section TEXT NOT NULL,
                    chapter TEXT,
                    page_numbers TEXT,
                    source_file TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding_count INTEGER DEFAULT 0
                )
            """)

            # Create document_chunks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id),
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section TEXT,
                    chapter TEXT,
                    page_numbers TEXT,
                    source_file TEXT,
                    embedding_id TEXT,  -- Reference to Qdrant embedding ID
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create document_questions table to track interactions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_questions (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    document_ids TEXT[],  -- Array of document IDs used to answer
                    selected_text_used BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info("Database tables initialized successfully")

    async def save_document_metadata(self, metadata: DocumentMetadata):
        """Save or update document metadata in the database."""
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO documents (
                        id, title, section, chapter, page_numbers, source_file, embedding_count
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        section = EXCLUDED.section,
                        chapter = EXCLUDED.chapter,
                        page_numbers = EXCLUDED.page_numbers,
                        source_file = EXCLUDED.source_file,
                        embedding_count = EXCLUDED.embedding_count,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    metadata.id,
                    metadata.title,
                    metadata.section,
                    metadata.chapter,
                    metadata.page_numbers,
                    metadata.source_file,
                    metadata.embedding_count
                )

                logger.info(f"Saved document metadata for ID: {metadata.id}")
            except Exception as e:
                logger.error(f"Error saving document metadata: {str(e)}")
                raise

    async def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata by ID."""
        async with self.connection_pool.acquire() as conn:
            try:
                row = await conn.fetchrow("""
                    SELECT id, title, section, chapter, page_numbers, source_file,
                           created_at, updated_at, embedding_count
                    FROM documents WHERE id = $1
                """, doc_id)

                if row:
                    return DocumentMetadata(
                        id=row['id'],
                        title=row['title'],
                        section=row['section'],
                        chapter=row['chapter'],
                        page_numbers=row['page_numbers'],
                        source_file=row['source_file'],
                        created_at=str(row['created_at']),
                        updated_at=str(row['updated_at']),
                        embedding_count=row['embedding_count']
                    )
                return None
            except Exception as e:
                logger.error(f"Error retrieving document metadata: {str(e)}")
                raise

    async def save_document_chunk(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        chunk_index: int,
        section: str,
        chapter: str = "",
        page_numbers: str = "",
        source_file: str = "",
        embedding_id: str = ""
    ):
        """Save a document chunk to the database."""
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO document_chunks (
                        id, document_id, content, chunk_index, section, chapter,
                        page_numbers, source_file, embedding_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        chunk_index = EXCLUDED.chunk_index,
                        section = EXCLUDED.section,
                        chapter = EXCLUDED.chapter,
                        page_numbers = EXCLUDED.page_numbers,
                        source_file = EXCLUDED.source_file,
                        embedding_id = EXCLUDED.embedding_id,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    chunk_id,
                    document_id,
                    content,
                    chunk_index,
                    section,
                    chapter,
                    page_numbers,
                    source_file,
                    embedding_id
                )

                logger.info(f"Saved document chunk: {chunk_id}")
            except Exception as e:
                logger.error(f"Error saving document chunk: {str(e)}")
                raise

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document."""
        async with self.connection_pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT id, content, chunk_index, section, chapter,
                           page_numbers, source_file, embedding_id, created_at
                    FROM document_chunks
                    WHERE document_id = $1
                    ORDER BY chunk_index
                """, document_id)

                chunks = []
                for row in rows:
                    chunks.append({
                        'id': row['id'],
                        'content': row['content'],
                        'chunk_index': row['chunk_index'],
                        'section': row['section'],
                        'chapter': row['chapter'],
                        'page_numbers': row['page_numbers'],
                        'source_file': row['source_file'],
                        'embedding_id': row['embedding_id'],
                        'created_at': str(row['created_at'])
                    })

                return chunks
            except Exception as e:
                logger.error(f"Error retrieving document chunks: {str(e)}")
                raise

    async def save_question_answer(
        self,
        question: str,
        answer: str,
        document_ids: List[str],
        selected_text_used: bool
    ):
        """Save a question-answer interaction to track usage."""
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO document_questions (
                        question, answer, document_ids, selected_text_used
                    ) VALUES ($1, $2, $3, $4)
                """,
                    question,
                    answer,
                    document_ids,
                    selected_text_used
                )

                logger.info(f"Saved question-answer interaction: {question[:50]}...")
            except Exception as e:
                logger.error(f"Error saving question-answer interaction: {str(e)}")
                raise

    async def get_recent_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent question-answer interactions."""
        async with self.connection_pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT question, answer, document_ids, selected_text_used, created_at
                    FROM document_questions
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)

                questions = []
                for row in rows:
                    questions.append({
                        'question': row['question'],
                        'answer': row['answer'],
                        'document_ids': row['document_ids'],
                        'selected_text_used': row['selected_text_used'],
                        'created_at': str(row['created_at'])
                    })

                return questions
            except Exception as e:
                logger.error(f"Error retrieving recent questions: {str(e)}")
                raise

    async def close(self):
        """Close the database connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Database connection pool closed")


# Global database manager instance
db_manager = DatabaseManager()


async def get_database_connection():
    """Dependency for FastAPI to get database connection."""
    if not db_manager.connection_pool:
        await db_manager.connect()
    return db_manager