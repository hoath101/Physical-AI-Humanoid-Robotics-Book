from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
import logging

from api.models.request import IngestionRequest
from api.models.response import IngestionResponse, ErrorResponse
from api.services.ingestion import ingestion_service
from api.middleware.auth import api_key_auth

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest book content",
    description="Ingest and index book content for RAG operations"
)
async def ingest_book(
    request: IngestionRequest,
    credentials=Depends(api_key_auth)
):
    """
    Endpoint to ingest book content into the system.
    """
    try:
        logger.info(f"Starting ingestion for book: {request.book_id}")

        # Call the ingestion service
        result = await ingestion_service.ingest_book_content(request)

        logger.info(f"Ingestion completed successfully for book: {request.book_id}")
        return result

    except ValueError as ve:
        logger.error(f"Validation error during ingestion: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Validation error: {str(ve)}",
                "book_id": request.book_id if 'request' in locals() else None
            }
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Ingestion failed: {str(e)}"
            }
        )


@router.delete(
    "/ingest/{book_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete book content",
    description="Delete all content associated with a specific book ID"
)
async def delete_book(
    book_id: str,
    credentials=Depends(api_key_auth)
):
    """
    Endpoint to delete book content from the system.
    """
    try:
        logger.info(f"Starting deletion for book: {book_id}")

        # Call the ingestion service to delete the book
        result = await ingestion_service.delete_book_content(book_id)

        logger.info(f"Deletion completed successfully for book: {book_id}")
        return result

    except Exception as e:
        logger.error(f"Error during deletion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Deletion failed: {str(e)}"
            }
        )


@router.put(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Update book content",
    description="Update existing book content by replacing old content with new content"
)
async def update_book(
    request: IngestionRequest,
    credentials=Depends(api_key_auth)
):
    """
    Endpoint to update existing book content.
    """
    try:
        logger.info(f"Starting update for book: {request.book_id}")

        # Call the ingestion service to update the book
        result = await ingestion_service.update_book_content(request)

        logger.info(f"Update completed successfully for book: {request.book_id}")
        return result

    except ValueError as ve:
        logger.error(f"Validation error during update: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Validation error: {str(ve)}",
                "book_id": request.book_id if 'request' in locals() else None
            }
        )
    except Exception as e:
        logger.error(f"Error during update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Update failed: {str(e)}"
            }
        )