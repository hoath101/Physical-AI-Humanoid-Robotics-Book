from fastapi import APIRouter
from .routes import ingest, query, health

# Create API router for version 1
router = APIRouter()

# Include all route modules
router.include_router(ingest.router, tags=["ingestion"])
router.include_router(query.router, tags=["query"])
router.include_router(health.router, tags=["health"])