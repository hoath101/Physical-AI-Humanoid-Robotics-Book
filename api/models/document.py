from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, DateTime, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class QueryMode(str, Enum):
    GLOBAL = "global"
    SELECTION_ONLY = "selection_only"

class BookContent(BaseModel):
    """
    Model representing a chunk of book content with metadata.
    """
    id: str = Field(..., description="Unique identifier for the content segment")
    text: str = Field(..., description="The actual content text")
    chapter: Optional[str] = Field(None, description="Chapter identifier/name")
    page: Optional[int] = Field(None, description="Page number")
    section: Optional[str] = Field(None, description="Section identifier")
    paragraph_id: Optional[str] = Field(None, description="Paragraph identifier within the section")
    book_id: str = Field(..., description="Reference to the book this content belongs to")
    embedding_vector: Optional[List[float]] = Field(None, description="Vector representation for similarity search")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (author, date, etc.)")

class Query(BaseModel):
    """
    Model representing a user query.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the query")
    question: str = Field(..., description="The user's question text")
    query_mode: QueryMode = Field(..., description="Enum (GLOBAL | SELECTION_ONLY)")
    selected_text: Optional[str] = Field(None, description="Text selected by user (for selection-only mode)")
    user_id: Optional[str] = Field(None, description="Reference to the user making the query (optional)")
    timestamp: Optional[datetime] = Field(None, description="When the query was made")
    session_id: Optional[str] = Field(None, description="Reference to the chat session")

class RetrievedContext(BaseModel):
    """
    Model representing retrieved context based on a query.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the retrieved context")
    content_segments: List[BookContent] = Field(..., description="Array of relevant book content segments")
    relevance_scores: List[float] = Field(..., description="Array of similarity scores for each segment")
    query_id: Optional[str] = Field(None, description="Reference to the original query")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional retrieval metadata")

class Response(BaseModel):
    """
    Model representing a response to a query.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the response")
    answer: str = Field(..., description="The generated answer text")
    citations: List[Dict[str, Any]] = Field(..., description="Array of source references with chapter/page/section info")
    query_id: Optional[str] = Field(None, description="Reference to the original query")
    timestamp: Optional[datetime] = Field(None, description="When the response was generated")
    grounding_confidence: Optional[float] = Field(None, description="Confidence level in the grounding (0-1)")

class SessionBase(BaseModel):
    """
    Base model representing a chat session.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the session")
    user_id: Optional[str] = Field(None, description="Reference to the user (optional for anonymous sessions)")
    book_id: Optional[str] = Field(None, description="Reference to the book for this session")
    start_time: Optional[datetime] = Field(None, description="When the session started")
    last_activity: Optional[datetime] = Field(None, description="When the last interaction occurred")
    active: bool = Field(True, description="Boolean indicating if session is currently active")

# SQLAlchemy model for database persistence
class SessionDB(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    book_id = Column(String, index=True)
    start_time = Column(DateTime)
    last_activity = Column(DateTime)
    active = Column(Boolean, default=True)

class Session(SessionBase):
    """
    Model representing a chat session (combines Pydantic and SQLAlchemy).
    """
    pass