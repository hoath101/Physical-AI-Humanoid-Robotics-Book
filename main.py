"""
Main application file for the Physical AI & Humanoid Robotics Book RAG Chatbot.
This file integrates all components: backend API, ingestion pipeline, and database.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Define the models that were previously in api.fastapi_app
class ChatRequest(BaseModel):
    question: str
    selected_text: Optional[str] = None
    book_section: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    mode_used: str  # "selected_text_only" or "full_book_qa"

from rag.ingestion import run_ingestion_pipeline, BookIngestionPipeline
from rag.retriever import RAGRetriever
from db.database import db_manager
from config.ingestion_config import get_config_value

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global components
rag_retriever: Optional[RAGRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize and cleanup resources."""
    global rag_retriever

    print("🚀 Starting Physical AI & Humanoid Robotics Book RAG Chatbot...")

    # Initialize components
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Initialize RAG retriever
    rag_retriever = RAGRetriever(openai_client)

    # Initialize Qdrant collection
    await rag_retriever.initialize_collection()

    # Connect to database
    await db_manager.connect()

    print("✅ All components initialized successfully")

    yield  # Run the application

    # Cleanup
    await db_manager.close()
    print("🛑 RAG Chatbot system shut down")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Physical AI & Humanoid Robotics Book RAG Chatbot",
    description="An AI-powered chatbot for answering questions about Physical AI and Humanoid Robotics using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Include the chat endpoint from the original API
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint for processing user questions.

    If selected_text is provided, uses selected-text-only mode.
    Otherwise, performs full-book QA with vector search.
    """
    try:
        from api.fastapi_app import ChatResponse
        from rag.retriever import RAGRetriever
        from db.database import db_manager

        global rag_retriever, openai_client

        # Input validation
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        if len(request.question) > 2000:  # Reasonable limit for questions
            raise HTTPException(status_code=400, detail="Question is too long (max 2000 characters)")

        if request.selected_text and len(request.selected_text) > 5000:  # Reasonable limit for selected text
            raise HTTPException(status_code=400, detail="Selected text is too long (max 5000 characters)")

        # Determine which mode to use based on presence of selected_text
        if request.selected_text and request.selected_text.strip():
            # Selected-text-only mode: Only use provided text
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert assistant for the 'Physical AI & Humanoid Robotics' book. "
                        "Answer the user's question using ONLY the provided selected text. "
                        "Do not use any external knowledge or make assumptions beyond what's in the selected text. "
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
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=30  # Add timeout to prevent hanging requests
                )

                answer = response.choices[0].message.content

                # Save the interaction to database
                await db_manager.save_question_answer(
                    question=request.question,
                    answer=answer,
                    document_ids=[],
                    selected_text_used=True
                )

                return ChatResponse(
                    answer=answer,
                    sources=None,  # No sources in selected-text mode
                    mode_used="selected_text_only"
                )
            except Exception as e:
                print(f"Error with OpenAI API in selected-text mode: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
        else:
            # Full-book QA mode: Use vector search and RAG
            global rag_retriever

            try:
                # Retrieve relevant chunks from the book
                retrieved_chunks = await rag_retriever.retrieve_relevant_chunks(
                    request.question,
                    book_section=request.book_section
                )
            except Exception as e:
                print(f"Error retrieving chunks from Qdrant: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error retrieving information from the book: {str(e)}")

            if not retrieved_chunks:
                # If no relevant chunks found, return appropriate response
                answer = (
                    "I couldn't find specific information in the book related to your question: "
                    f"{request.question}. Please check if your question is covered in the book content."
                )

                # Save the interaction to database
                await db_manager.save_question_answer(
                    question=request.question,
                    answer=answer,
                    document_ids=[],
                    selected_text_used=False
                )

                return ChatResponse(
                    answer=answer,
                    sources=[],
                    mode_used="full_book_qa"
                )

            # Prepare context for the LLM from retrieved chunks
            context_str = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])
            sources = [chunk['metadata'] for chunk in retrieved_chunks]
            document_ids = [chunk['metadata']['id'] for chunk in retrieved_chunks]

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
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=30  # Add timeout to prevent hanging requests
                )

                answer = response.choices[0].message.content

                # Save the interaction to database
                await db_manager.save_question_answer(
                    question=request.question,
                    answer=answer,
                    document_ids=document_ids,
                    selected_text_used=False
                )

                return ChatResponse(
                    answer=answer,
                    sources=sources,
                    mode_used="full_book_qa"
                )
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
async def ingest_books(request: IngestionRequest, background_tasks: BackgroundTasks):
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

        # Check if OpenAI client is initialized
        if not openai_client.api_key:
            raise HTTPException(status_code=500, detail="OpenAI client not properly initialized")

        # Run ingestion in background
        background_tasks.add_task(
            run_ingestion_pipeline,
            request.book_directory,
            openai_client,
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
async def test_query_endpoint(test_query: TestQuery):
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

            selected_text_result = await chat_endpoint(ChatRequest(
                question=test_query.question,
                selected_text=sample_text
            ))
            results["selected_text_only"] = selected_text_result

        if test_query.mode in ["both", "full_book_qa"]:
            # Test full-book QA mode
            full_book_result = await chat_endpoint(ChatRequest(
                question=test_query.question,
                selected_text=None
            ))
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


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "component": "full-system"}


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
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/info", "method": "GET", "description": "System information"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "rag", "db", "config"]
    )