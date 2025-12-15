from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
import logging

from api.models.request import QueryRequest
from api.models.response import QueryResponse, ErrorResponse
from api.services.chat import chat_service
from api.middleware.auth import api_key_auth

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query book content (Global mode)",
    description="Query the entire book corpus for relevant information"
)
async def query_book_global(
    request: QueryRequest,
    credentials=Depends(api_key_auth)
):
    """
    Endpoint to query book content in global mode (entire book corpus).
    """
    try:
        logger.info(f"Processing global query for book: {request.book_id}")

        # Ensure the query mode is global for this endpoint
        if request.query_mode.value != "global":
            # Override to global since this is the global query endpoint
            request.query_mode = type(request.query_mode).GLOBAL

        # Get session_id from headers or request, default to None if not provided
        session_id = request.headers.get("X-Session-ID")  # This would come from the header in a real implementation
        user_id = request.headers.get("X-User-ID")  # This would come from the header in a real implementation

        # Call the chat service to generate response
        result = await chat_service.generate_response(
            query=request.question,
            book_id=request.book_id,
            query_mode=request.query_mode,
            selected_text=request.selected_text,
            session_id=session_id,
            user_id=user_id
        )

        logger.info(f"Global query completed successfully for book: {request.book_id}")
        return result

    except ValueError as ve:
        logger.error(f"Validation error during global query: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Validation error: {str(ve)}",
                "book_id": request.book_id if 'request' in locals() else None
            }
        )
    except Exception as e:
        logger.error(f"Error during global query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Query failed: {str(e)}"
            }
        )


@router.post(
    "/query/selection",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query book content (Selection-only mode)",
    description="Query only the selected text for relevant information"
)
async def query_book_selection(
    request: QueryRequest,
    credentials=Depends(api_key_auth)
):
    """
    Endpoint to query book content in selection-only mode (only selected text).
    """
    try:
        logger.info(f"Processing selection-only query for book: {request.book_id}")

        # Ensure the query mode is selection_only for this endpoint
        request.query_mode = type(request.query_mode).SELECTION_ONLY

        # Validate that selected_text is provided
        if not request.selected_text or not request.selected_text.strip():
            raise ValueError("Selected text is required for selection-only mode")

        # Get session_id from headers or request, default to None if not provided
        session_id = request.headers.get("X-Session-ID")  # This would come from the header in a real implementation
        user_id = request.headers.get("X-User-ID")  # This would come from the header in a real implementation

        # Call the chat service to generate response
        result = await chat_service.generate_response(
            query=request.question,
            book_id=request.book_id,
            query_mode=request.query_mode,
            selected_text=request.selected_text,
            session_id=session_id,
            user_id=user_id
        )

        logger.info(f"Selection-only query completed successfully for book: {request.book_id}")
        return result

    except ValueError as ve:
        logger.error(f"Validation error during selection query: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Validation error: {str(ve)}",
                "book_id": request.book_id if 'request' in locals() else None
            }
        )
    except Exception as e:
        logger.error(f"Error during selection query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Query failed: {str(e)}"
            }
        )


# Additional endpoint that supports both modes based on the request
@router.post(
    "/query/mode",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query book content (Flexible mode)",
    description="Query book content supporting both global and selection-only modes"
)
async def query_book_flexible(
    request: QueryRequest,
    credentials=Depends(api_key_auth)
):
    """
    Flexible endpoint that supports both global and selection-only query modes.
    """
    try:
        logger.info(f"Processing flexible query (mode: {request.query_mode.value}) for book: {request.book_id}")

        # Call the chat service to generate response based on mode
        result = await chat_service.generate_response(
            query=request.question,
            book_id=request.book_id,
            query_mode=request.query_mode,
            selected_text=request.selected_text,
            session_id=None  # In a real implementation, you'd get this from the request
        )

        logger.info(f"Flexible query completed successfully for book: {request.book_id}, mode: {request.query_mode.value}")
        return result

    except ValueError as ve:
        logger.error(f"Validation error during flexible query: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Validation error: {str(ve)}",
                "book_id": request.book_id if 'request' in locals() else None
            }
        )
    except Exception as e:
        logger.error(f"Error during flexible query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Query failed: {str(e)}"
            }
        )