from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from .request import Citation, QueryMode

class StatusEnum(str, Enum):
    SUCCESS = "success"
    NO_CONTENT_FOUND = "no_content_found"
    ERROR = "error"

class IngestionResponse(BaseModel):
    """
    Response model for book content ingestion.
    """
    status: StatusEnum = Field(StatusEnum.SUCCESS, description="Status of the operation")
    message: str = Field(..., description="Human-readable message about the result")
    book_id: str = Field(..., description="Identifier for the ingested book")
    chunks_processed: int = Field(..., description="Number of content chunks processed")
    processing_time: str = Field(..., description="Time taken to process the content")

class QueryResponse(BaseModel):
    """
    Response model for querying book content.
    """
    answer: str = Field(..., description="The answer to the question")
    citations: List[Citation] = Field(..., description="Array of source citations with location information")
    query_mode: QueryMode = Field(..., description="The mode used for this query")
    retrieved_context_count: int = Field(..., description="Number of context items retrieved")
    status: Optional[StatusEnum] = Field(StatusEnum.SUCCESS, description="Status of the query")
    processing_time: Optional[str] = Field(None, description="Time taken to process the query")

class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    """
    status: StatusEnum = Field(StatusEnum.ERROR, description="Status of the operation")
    message: str = Field(..., description="Human-readable error message")
    error_id: Optional[str] = Field(None, description="Unique identifier for the error")

class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Health status of the service")
    timestamp: str = Field(..., description="Timestamp of the health check")
    version: str = Field(..., description="Version of the service")
    dependencies: dict = Field(..., description="Status of dependent services")