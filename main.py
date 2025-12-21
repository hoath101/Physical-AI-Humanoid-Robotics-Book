"""
Main application file for the Physical AI & Humanoid Robotics Book RAG Chatbot.
This file integrates all components: backend API, ingestion pipeline, and database.
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv
from middleware.auth import authenticate_request, check_rate_limit, security
from services.cache import init_cache, close_cache
import services.cache as cache_module
from services.ai_client import AIClient

from rag.ingestion import run_ingestion_pipeline, BookIngestionPipeline
from rag.retriever import RAGRetriever
from db.database import db_manager
from config.ingestion_config import get_config_value
from services.chat_adapter import init_chat_adapter, get_chat_adapter, Message, Thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the models that match the frontend's expected format
class QueryMode(str, Enum):
    GLOBAL = "global"
    SELECTION_ONLY = "selection_only"

class QueryRequest(BaseModel):
    question: str
    query_mode: QueryMode
    book_id: str
    selected_text: Optional[str] = None
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.3

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    query_mode: QueryMode
    retrieved_context_count: int
    status: Optional[str] = "success"
    processing_time: Optional[str] = None



# Load environment variables
load_dotenv()

# Global components
ai_client: Optional[AIClient] = None
rag_retriever: Optional[RAGRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize and cleanup resources."""
    global ai_client, rag_retriever

    print("Starting Physical AI & Humanoid Robotics Book RAG Chatbot...")

    # Initialize AI client
    ai_client = AIClient()
    if not ai_client.api_key:
        raise ValueError("AI API KEY environment variable is not set")

    # Initialize RAG retriever
    rag_retriever = RAGRetriever(ai_client)

    # Initialize Qdrant collection
    try:
        await rag_retriever.initialize_collection()
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}")
        logger.warning("Qdrant not available - search functionality will be disabled until Qdrant is running")

    # Initialize chat adapter
    try:
        await init_chat_adapter(ai_client, rag_retriever)
        logger.info("Chat adapter initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing chat adapter: {e}")
        raise

    # Connect to database (optional - app can run without database for basic functionality)
    try:
        await db_manager.connect()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.warning("App will run without database functionality (question logging will be disabled)")

    # Initialize cache
    await init_cache()

    print("All components initialized successfully")

    yield  # Run the application

    # Cleanup
    await db_manager.close()
    await close_cache()
    print("🛑 RAG Chatbot system shut down")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Physical AI & Humanoid Robotics Book RAG Chatbot",
    description="An AI-powered chatbot for answering questions about Physical AI and Humanoid Robotics using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from Docusaurus frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hoath101.github.io",
        "http://localhost:3000",
        "http://localhost:3001",
        "https://huggingface.co"
    ],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose headers that might be needed by the frontend
    expose_headers=["Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"]
)

# Include the chat endpoint from the original API
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(
    request: QueryRequest,
    api_key: str = Depends(authenticate_request),
    rate_limit: str = Depends(check_rate_limit)
):
    """
    Main endpoint for processing user questions using RAG.

    Supports both global (full book) and selection-only modes.
    """
    print(f"\n{'='*60}")
    print(f"[CHAT] Incoming chat request:")
    print(f"       Question: {request.question[:100]}...")
    print(f"       Query Mode: {request.query_mode}")
    print(f"       Book ID: {request.book_id}")
    print(f"{'='*60}\n")

    try:
        from rag.retriever import RAGRetriever
        from db.database import db_manager

        global rag_retriever, ai_client

        # Input validation
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        if len(request.question) > 2000:  # Reasonable limit for questions
            raise HTTPException(status_code=400, detail="Question is too long (max 2000 characters)")

        if request.selected_text and len(request.selected_text) > 5000:  # Reasonable limit for selected text
            raise HTTPException(status_code=400, detail="Selected text is too long (max 5000 characters)")

        # Check cache first for this query
        cached_response = await cache_module.cache_service.get_query_response(
            request.question,
            request.selected_text,
            request.book_id
        )

        if cached_response:
            print(f"Cache hit for question: {request.question[:50]}...")
            # Convert cached response to match expected format
            return QueryResponse(
                answer=cached_response.get('answer', ''),
                citations=cached_response.get('citations', []),
                query_mode=QueryMode(cached_response.get('query_mode', 'global')),
                retrieved_context_count=cached_response.get('retrieved_context_count', 0),
                status=cached_response.get('status', 'success')
            )

        # Determine which mode to use based on query_mode
        if request.query_mode == QueryMode.SELECTION_ONLY and request.selected_text and request.selected_text.strip():
            print("[MODE] Using SELECTION_ONLY mode")
            # Selected-text-only mode: Only use provided text
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert assistant for the 'Physical AI & Humanoid Robotics' book. "
                        "Answer the user's question using ONLY the provided selected text. "
                        "Do not use any external knowledge or make assumptions beyond what's in the provided text. "
                        "If the answer cannot be found in the provided text, clearly state that the information is not available in the selected text."
                        "Be concise and accurate, and always cite information from the provided text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {request.question}\n\nSelected Text: {request.selected_text}"
                }
            ]

            try:
                answer = await ai_client.create_chat_completion(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=1000
                )

                # Save the interaction to database
                await db_manager.save_question_answer(
                    question=request.question,
                    answer=answer,
                    document_ids=[],
                    selected_text_used=True
                )

                # Create response object with empty citations for selection-only mode
                response_obj = QueryResponse(
                    answer=answer,
                    citations=[],  # No citations in selection-only mode
                    query_mode=request.query_mode,
                    retrieved_context_count=0,
                    status="success"
                )

                # Cache the response for future queries
                await cache_module.cache_service.set_query_response(
                    request.question,
                    request.selected_text,
                    request.book_id,
                    response_obj.model_dump()
                )

                return response_obj
            except Exception as e:
                print(f"Error with OpenAI API in selected-text mode: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
        else:
            print("[MODE] Using GLOBAL mode (RAG with vector search)")
            # Global mode: Use vector search and RAG
            global rag_retriever

            try:
                print(f"[RAG] Retrieving relevant chunks from vector database...")
                # Retrieve relevant chunks from the book
                retrieved_chunks = await rag_retriever.retrieve_relevant_chunks(
                    request.question,
                    book_section=None  # We'll search across all sections
                )
                print(f"[RAG] Retrieved {len(retrieved_chunks)} relevant chunks")
            except Exception as e:
                print(f"[ERROR] Error retrieving chunks from Qdrant: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error retrieving information from the book: {str(e)}")

            if not retrieved_chunks:
                print("[WARN] No relevant chunks found for the query")
                # If no relevant chunks found, return appropriate response
                answer = (
                    "I couldn't find specific information in the book related to your question: "
                    f"{request.question}. Please check if your question is covered in the book content."
                )

                # Create response object
                response_obj = QueryResponse(
                    answer=answer,
                    citations=[],
                    query_mode=request.query_mode,
                    retrieved_context_count=0,
                    status="no_content_found"
                )

                # Save the interaction to database
                await db_manager.save_question_answer(
                    question=request.question,
                    answer=answer,
                    document_ids=[],
                    selected_text_used=False
                )

                # Cache the response for future queries
                await cache_module.cache_service.set_query_response(
                    request.question,
                    request.selected_text,
                    request.book_id,
                    response_obj.model_dump()
                )

                return response_obj

            # Prepare context for the LLM from retrieved chunks
            context_str = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])

            # Create citation objects from the retrieved chunks
            citations = []
            for chunk in retrieved_chunks:
                metadata = chunk['metadata']
                citation = {
                    "chapter": metadata.get('chapter', 'Unknown'),
                    "section": metadata.get('section', 'Unknown'),
                    "page": metadata.get('page_numbers', None),
                    "paragraph_id": metadata.get('id', ''),
                    "text_snippet": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                }
                citations.append(citation)

            document_ids = [chunk['id'] for chunk in retrieved_chunks]

            # Check if the context is too long for the model
            max_context_length = 10000  # Approximate limit for model context
            if len(context_str) > max_context_length:
                # Truncate context to fit in model limits
                context_str = context_str[:max_context_length]
                print(f"Context was too long and was truncated to {max_context_length} characters")

            # Format prompt for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert assistant for the 'Physical AI & Humanoid Robotics' book. "
                        "Answer the user's question using ONLY the provided context from the book. "
                        "Reference specific chapters or sections mentioned in the context when possible. "
                        "If the answer cannot be found in the provided context, clearly state that the information is not available in the provided book excerpts."
                        "Be concise and accurate, and always cite information from the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {request.question}\n\nContext from the book:\n{context_str}"
                }
            ]

            try:
                print(f"[AI] Calling AI API to generate response...")
                answer = await ai_client.create_chat_completion(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=1000
                )
                print(f"[AI] Generated response successfully")

                # Save the interaction to database
                await db_manager.save_question_answer(
                    question=request.question,
                    answer=answer,
                    document_ids=document_ids,
                    selected_text_used=False
                )

                # Create response object
                response_obj = QueryResponse(
                    answer=answer,
                    citations=citations,
                    query_mode=request.query_mode,
                    retrieved_context_count=len(retrieved_chunks),
                    status="success"
                )

                # Cache the response for future queries
                await cache_module.cache_service.set_query_response(
                    request.question,
                    request.selected_text,
                    request.book_id,
                    response_obj.model_dump()
                )

                return response_obj
            except Exception as e:
                print(f"Error with OpenAI API in full-book QA mode: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        print(f"Unexpected error processing chat request: {str(e)}")
        import traceback
        traceback.print_exc()  # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Endpoint for running ingestion
class IngestionRequest(BaseModel):
    book_directory: str
    max_chunk_size: int = 1000
    overlap: int = 100


@app.post("/ingest")
async def ingest_books(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(authenticate_request)
):
    """
    Endpoint to trigger the book ingestion pipeline.
    This runs the ingestion in the background to avoid blocking the API.
    """
    try:
        # Validate inputs
        if not request.book_directory or len(request.book_directory.strip()) == 0:
            raise HTTPException(status_code=400, detail="Book directory cannot be empty")

        if not os.path.exists(request.book_directory):
            raise HTTPException(status_code=400, detail=f"Directory does not exist: {request.book_directory}")

        # Validate chunk size and overlap parameters
        if request.max_chunk_size <= 0 or request.max_chunk_size > 5000:
            raise HTTPException(status_code=400, detail="max_chunk_size must be between 1 and 5000 characters")

        if request.overlap < 0 or request.overlap >= request.max_chunk_size:
            raise HTTPException(status_code=400, detail="overlap must be between 0 and max_chunk_size-1")

        # Check if AI client is initialized
        if not ai_client.api_key:
            raise HTTPException(status_code=500, detail="AI client not properly initialized")

        # Run ingestion in background with error handling
        background_tasks.add_task(
            safe_run_ingestion_pipeline,
            request.book_directory,
            ai_client,
            request.max_chunk_size,
            request.overlap
        )

        return {
            "message": "Ingestion started successfully",
            "directory": request.book_directory,
            "max_chunk_size": request.max_chunk_size,
            "overlap": request.overlap
        }
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        print(f"Error starting ingestion: {str(e)}")
        import traceback
        traceback.print_exc()  # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Error starting ingestion: {str(e)}")


# Endpoint for getting ingestion status
@app.get("/ingestion-status")
async def ingestion_status():
    """
    Endpoint to get the current status of the system.
    """
    global rag_retriever

    if rag_retriever is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        chunk_count = await rag_retriever.get_chunk_count()
        sections = await rag_retriever.get_all_sections()

        return {
            "status": "ready",
            "vector_db_status": "connected",
            "chunk_count": chunk_count,
            "indexed_sections": sections,
            "indexed_section_count": len(sections)
        }
    except Exception as e:
        print(f"Error getting status: {str(e)}")
        import traceback
        traceback.print_exc()  # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")


# Endpoint for testing the system with sample queries
class TestQuery(BaseModel):
    question: str
    mode: str = "both"  # "selected_text_only", "full_book_qa", or "both"


@app.post("/test-query")
async def test_query_endpoint(
    test_query: TestQuery,
    api_key: str = Depends(authenticate_request),
    rate_limit: str = Depends(check_rate_limit)
):
    """
    Endpoint to test the system with sample queries in different modes.
    Useful for validation and testing.
    """
    try:
        results = {}

        if test_query.mode in ["both", "selected_text_only"]:
            # Test selected-text-only mode with a sample text
            sample_text = (
                "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. "
                "It provides a collection of tools, libraries, and conventions that aim to simplify "
                "the task of creating complex and robust robot behavior across a wide variety of robot platforms. "
                "Key features include improved support for real-time systems, better tools for distributed systems, "
                "and enhanced security."
            )

            selected_text_request = QueryRequest(
                question=test_query.question,
                query_mode=QueryMode.SELECTION_ONLY,
                book_id="test-book",
                selected_text=sample_text
            )
            selected_text_result = await chat_endpoint(selected_text_request)
            results["selected_text_only"] = selected_text_result

        if test_query.mode in ["both", "full_book_qa"]:
            # Test full-book QA mode
            full_book_request = QueryRequest(
                question=test_query.question,
                query_mode=QueryMode.GLOBAL,
                book_id="test-book"
            )
            full_book_result = await chat_endpoint(full_book_request)
            results["full_book_qa"] = full_book_result

        return {
            "question": test_query.question,
            "mode": test_query.mode,
            "results": results
        }
    except Exception as e:
        print(f"Error testing query: {str(e)}")
        import traceback
        traceback.print_exc()  # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Error testing query: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Physical AI & Humanoid Robotics Book API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
            "collection_info": "/collection-info"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "component": "full-system"}


# Configuration endpoint - provides runtime configuration to the frontend
# This does NOT expose sensitive data like API keys
@app.get("/config")
async def get_runtime_config():
    """Get runtime configuration for the frontend"""
    return {
        "chatbot": {
            "enabled": True,
            "title": "Book Assistant",
            "initialOpen": False,
            # Note: apiUrl is provided here but should be configured based on deployment
            # In production, this should match the backend API URL
            "apiUrl": os.getenv("FRONTEND_API_URL", "http://localhost:8000"),
            "bookId": os.getenv("DEFAULT_BOOK_ID", "default-book"),
        }
    }


# ChatKit-compatible session endpoint - creates a session for the chat adapter
@app.post("/api/chatkit/session")
async def create_chatkit_session():
    """Create a new session and return a client secret for the chat adapter"""
    try:
        import secrets
        import time

        # Create a session token (in a real implementation, this would be more sophisticated)
        client_secret = f"chat_adapter_token_{secrets.token_urlsafe(16)}"

        # Return the client secret
        return {
            "client_secret": client_secret,
            "expires_at": int(time.time()) + 3600,  # Expires in 1 hour
            "user_id": "user_" + secrets.token_urlsafe(8)  # Generate a user ID
        }
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating chat session: {str(e)}")


# ChatKit-compatible refresh endpoint - refreshes an existing session
@app.post("/api/chatkit/refresh")
async def refresh_chatkit_session(request: Request):
    """Refresh an existing session and return a new client secret"""
    try:
        import secrets
        import time

        # Get the request body
        body = await request.json()
        existing_token = body.get('token')

        # Create a new session token
        client_secret = f"chat_adapter_token_{secrets.token_urlsafe(16)}"

        # Return the new client secret
        return {
            "client_secret": client_secret,
            "expires_at": int(time.time()) + 3600,  # Expires in 1 hour
            "user_id": "user_" + secrets.token_urlsafe(8)  # Generate a user ID
        }
    except Exception as e:
        logger.error(f"Error refreshing chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing chat session: {str(e)}")


# ChatKit-compatible API endpoints for the chat adapter
@app.post("/api/chatkit/users")
async def create_user(request: Request):
    """Create a user in the chat system"""
    try:
        body = await request.json()
        user_id = body.get('id', 'default_user')
        name = body.get('name', 'Anonymous User')

        chat_adapter = get_chat_adapter()
        user = await chat_adapter.create_user(user_id, name)

        return {
            "id": user.id,
            "name": user.name,
            "created_at": user.created_at
        }
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


@app.post("/api/chatkit/threads")
async def create_thread(request: Request):
    """Create a new thread in the chat system"""
    try:
        body = await request.json()
        name = body.get('name', 'New Discussion')

        chat_adapter = get_chat_adapter()
        thread = await chat_adapter.create_thread(name)

        return {
            "id": thread.id,
            "name": thread.name,
            "created_at": thread.created_at,
            "updated_at": thread.updated_at
        }
    except Exception as e:
        logger.error(f"Error creating thread: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating thread: {str(e)}")


@app.get("/api/chatkit/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get a specific thread"""
    try:
        chat_adapter = get_chat_adapter()
        thread = await chat_adapter.get_thread(thread_id)

        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        return {
            "id": thread.id,
            "name": thread.name,
            "messages": [msg.dict() for msg in thread.messages],
            "created_at": thread.created_at,
            "updated_at": thread.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting thread: {str(e)}")


@app.get("/api/chatkit/threads")
async def get_threads():
    """Get all threads"""
    try:
        chat_adapter = get_chat_adapter()
        threads = await chat_adapter.get_threads()

        return {
            "threads": [
                {
                    "id": thread.id,
                    "name": thread.name,
                    "created_at": thread.created_at,
                    "updated_at": thread.updated_at
                }
                for thread in threads
            ]
        }
    except Exception as e:
        logger.error(f"Error getting threads: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting threads: {str(e)}")


@app.post("/api/chatkit/threads/{thread_id}/messages")
async def send_message(thread_id: str, request: Request):
    """Send a message to a thread"""
    try:
        body = await request.json()
        text = body.get('text', '')
        user_id = body.get('sender_id', 'default_user')

        if not text:
            raise HTTPException(status_code=400, detail="Message text is required")

        chat_adapter = get_chat_adapter()
        response_message = await chat_adapter.send_message(thread_id, user_id, text)

        return {
            "id": response_message.id,
            "text": response_message.text,
            "sender_id": response_message.sender_id,
            "created_at": response_message.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")


@app.get("/api/chatkit/threads/{thread_id}/messages")
async def get_messages(thread_id: str, limit: int = 100):
    """Get messages from a thread"""
    try:
        chat_adapter = get_chat_adapter()
        messages = await chat_adapter.get_messages(thread_id, limit)

        return {
            "messages": [
                {
                    "id": msg.id,
                    "text": msg.text,
                    "sender_id": msg.sender_id,
                    "created_at": msg.created_at
                }
                for msg in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting messages: {str(e)}")


# Info endpoint
@app.get("/info")
async def get_info():
    """Get information about the API"""
    return {
        "title": "Physical AI & Humanoid Robotics Book RAG Chatbot",
        "version": "1.0.0",
        "modes": ["full_book_qa", "selected_text_only"],
        "description": "RAG-based question answering for the Physical AI & Humanoid Robotics educational book",
        "endpoints": [
            {"path": "/chat", "method": "POST", "description": "Main chat endpoint"},
            {"path": "/ingest", "method": "POST", "description": "Trigger book ingestion"},
            {"path": "/ingestion-status", "method": "GET", "description": "Get system status"},
            {"path": "/config", "method": "GET", "description": "Runtime configuration for frontend"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/info", "method": "GET", "description": "System information"}
        ]
    }


# Safe wrapper for ingestion pipeline to handle exceptions in background tasks
async def safe_run_ingestion_pipeline(
    book_directory: str,
    ai_client,
    max_chunk_size: int = 1000,
    overlap: int = 100
):
    """
    Safe wrapper for the ingestion pipeline that catches exceptions to prevent
    background task errors from affecting the ASGI application.
    """
    try:
        from rag.ingestion import run_ingestion_pipeline
        await run_ingestion_pipeline(
            book_directory=book_directory,
            ai_client=ai_client,
            max_chunk_size=max_chunk_size,
            overlap=overlap
        )
    except Exception as e:
        logger.error(f"Error in background ingestion task: {e}")
        # Log the error but don't re-raise to prevent ASGI application exceptions
        # The ingestion will fail gracefully with a warning in the logs


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "rag", "db", "config"]
    )