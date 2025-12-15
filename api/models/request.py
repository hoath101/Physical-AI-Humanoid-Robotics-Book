from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class QueryMode(str, Enum):
    GLOBAL = "global"
    SELECTION_ONLY = "selection_only"

class BookMetadata(BaseModel):
    """
    Model for book metadata.
    """
    title: str = Field(..., description="Title of the book")
    author: str = Field(..., description="Author of the book")
    isbn: Optional[str] = Field(None, description="ISBN of the book")
    edition: Optional[str] = Field(None, description="Edition of the book")
    publication_date: Optional[str] = Field(None, description="Publication date in ISO 8601 format")

class IngestionRequest(BaseModel):
    """
    Request model for ingesting book content.
    """
    book_content: str = Field(..., description="Full text content of the book")
    book_metadata: BookMetadata = Field(..., description="Metadata about the book")
    chunk_size: int = Field(1000, ge=100, le=5000, description="Size of text chunks for processing")
    overlap_size: int = Field(200, ge=0, le=1000, description="Overlap between chunks")
    book_id: str = Field(..., description="Unique identifier for the book")

class QueryRequest(BaseModel):
    """
    Request model for querying book content.
    """
    question: str = Field(..., description="The user's question")
    query_mode: QueryMode = Field(QueryMode.GLOBAL, description="Query mode: global or selection_only")
    book_id: str = Field(..., description="Identifier for the book to query")
    selected_text: Optional[str] = Field(None, description="Text selected by user (required for selection_only mode)")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    temperature: float = Field(0.3, ge=0.0, le=1.0, description="LLM temperature setting")

class Citation(BaseModel):
    """
    Model for citation information.
    """
    chapter: str
    page: Optional[int] = None
    section: Optional[str] = None
    paragraph_id: Optional[str] = None
    text_snippet: str