from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

def customize_openapi(app: FastAPI):
    """
    Customize the OpenAPI schema for the application.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="RAG Chatbot API",
        version="1.0.0",
        description="""
# RAG Chatbot API Documentation

This API provides a Retrieval-Augmented Generation (RAG) system for querying book content with high factual accuracy and strict grounding in retrieved text.

## Overview

The RAG Chatbot API enables users to ask questions about book content and receive accurate answers based on the book's text. The system supports two query modes:

- **Global Mode**: Answers using the entire book corpus
- **Selection-Only Mode**: Answers using only user-selected text

## Authentication

All API endpoints require authentication using an API key passed in the Authorization header:

```
Authorization: Bearer {your-api-key}
```

## Rate Limiting

The API implements rate limiting to prevent abuse. Default limits are 100 requests per minute per API key.

## Response Format

All responses follow a consistent structure with the following fields:
- `answer`: The AI-generated answer to the query
- `contexts`: Retrieved text passages that support the answer
- `citations`: References to the source material
        """,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": "API Key authentication. Format: `Bearer {your-api-key}`"
        }
    }

    # Apply security globally
    openapi_schema["security"] = [{"BearerAuth": []}]

    # Enhance path descriptions and examples
    for path, methods in openapi_schema["paths"].items():
        for method, details in methods.items():
            # Add more detailed descriptions for each endpoint
            if path == "/api/v1/ingest" and method == "post":
                details["summary"] = "Ingest Book Content"
                details["description"] = """
Ingest and index book content for retrieval. This endpoint processes the provided text,
chunks it appropriately, generates embeddings, and stores it in the vector database.

The content will be available for querying after successful ingestion.
                """
                details["requestBody"] = {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/IngestionRequest"
                            },
                            "example": {
                                "title": "Chapter 1: Introduction to AI",
                                "content": "Artificial intelligence is a branch of computer science...",
                                "book_id": "my-book-id",
                                "chapter_id": "ch1",
                                "page_number": 1,
                                "paragraph_id": "p1"
                            }
                        }
                    }
                }
            elif path == "/api/v1/query" and method == "post":
                details["summary"] = "Query Book Content"
                details["description"] = """
Query the book content using Retrieval-Augmented Generation. Supports two modes:

- **GLOBAL**: Search across the entire book corpus
- **SELECTION_ONLY**: Search only within the selected text provided

The response will include the answer, supporting contexts, and citations.
                """
                details["requestBody"] = {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/QueryRequest"
                            },
                            "example": {
                                "query": "What is artificial intelligence?",
                                "book_id": "my-book-id",
                                "mode": "GLOBAL"
                            }
                        }
                    }
                }
            elif path == "/api/v1/health" and method == "get":
                details["summary"] = "Health Check"
                details["description"] = "Check the health status of the API service."
            elif path == "/api/v1/health/detail" and method == "get":
                details["summary"] = "Detailed Health Check"
                details["description"] = "Get detailed health information about all system components."

            # Enhance response descriptions
            if "responses" in details:
                for code, response in details["responses"].items():
                    if code == "200":
                        if path == "/api/v1/ingest":
                            response["description"] = "Content successfully ingested"
                            response["content"] = {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/IngestionResponse"
                                    },
                                    "example": {
                                        "success": True,
                                        "document_id": "doc_abc123",
                                        "chunks_processed": 1
                                    }
                                }
                            }
                        elif path == "/api/v1/query":
                            response["description"] = "Query processed successfully"
                            response["content"] = {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/QueryResponse"
                                    },
                                    "example": {
                                        "answer": "Artificial intelligence is...",
                                        "contexts": [
                                            {
                                                "content": "Original text passage...",
                                                "score": 0.9,
                                                "metadata": {"chapter": "ch1", "page": 1}
                                            }
                                        ],
                                        "citations": [
                                            {
                                                "text": "Original text",
                                                "page": 1,
                                                "chapter": "ch1"
                                            }
                                        ]
                                    }
                                }
                            }
                        elif path in ["/api/v1/health", "/api/v1/health/detail"]:
                            response["description"] = "Health status retrieved successfully"
                    elif code == "400":
                        response["description"] = "Bad request - Invalid input parameters"
                        response["content"] = {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        }
                    elif code == "401":
                        response["description"] = "Unauthorized - Invalid or missing API key"
                        response["content"] = {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        }
                    elif code == "422":
                        response["description"] = "Unprocessable entity - Validation error"
                        response["content"] = {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        }
                    elif code == "429":
                        response["description"] = "Rate limit exceeded"
                        response["content"] = {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        }
                    elif code == "500":
                        response["description"] = "Internal server error"
                        response["content"] = {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ErrorResponse"
                                }
                            }
                        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Example usage in main.py:
"""
from api.docs.openapi import customize_openapi

# After creating the app
app = FastAPI(...)
customize_openapi(app)
"""