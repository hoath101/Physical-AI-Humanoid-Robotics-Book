from fastapi import APIRouter, status
from typing import Dict, Any
import logging
from datetime import datetime

from api.models.response import HealthResponse
from api.config.settings import settings
from api.config.vector_db import vector_db

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the API and its dependencies"
)
async def health_check():
    """
    Health check endpoint that verifies the status of the API and its dependencies.
    """
    # Check dependencies
    dependencies = {
        "qdrant": "disconnected",
        "postgres": "disconnected",  # This would be connected in a full implementation
        "openai": "not tested"      # This would be tested in a full implementation
    }

    try:
        # Test Qdrant connection by initializing the collection
        vector_db.initialize_collection()
        dependencies["qdrant"] = "connected"
    except Exception as e:
        logger.error(f"Qdrant connection failed: {str(e)}")
        dependencies["qdrant"] = "disconnected"

    # In a real implementation, you would also test:
    # - Postgres connection
    # - OpenAI API availability
    # - Other external services

    health_response = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "dependencies": dependencies
    }

    logger.info("Health check completed")
    return health_response

@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if the API is ready to handle requests"
)
async def readiness_check():
    """
    Readiness check endpoint to verify the API is ready to handle requests.
    """
    # In a real implementation, you would check if all required services are ready
    # For now, we'll just return a simple ready status
    return {"status": "ready", "timestamp": datetime.now().isoformat()}

@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if the API process is alive"
)
async def liveness_check():
    """
    Liveness check endpoint to verify the API process is alive.
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}