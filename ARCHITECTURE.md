# Project Architecture

This project contains two distinct implementations that serve different purposes:

## 1. Modern API Implementation (`api/` directory)

The `api/` directory contains a well-structured, modern FastAPI implementation with:
- Proper separation of concerns (models, services, routes, middleware)
- SQLAlchemy ORM for database operations
- Comprehensive request/response validation with Pydantic
- Proper authentication and rate limiting middleware
- Clean service layer architecture
- Full CRUD operations for RAG functionality

**Endpoints**: `/api/v1/` with sub-endpoints for ingestion, query, and health checks

## 2. Legacy Pipeline Implementation (`main.py`, `rag/`, `db/`)

The older implementation includes:
- Direct asyncpg database operations
- Standalone ingestion pipeline in `rag/` and `run_ingestion.py`
- `main.py` as the original application entry point
- Simpler, more direct implementation approach

## Current Architecture

- **Primary API**: The `api/` directory implementation should be used for new development
- **Ingestion Pipeline**: The `run_ingestion.py` script and `rag/` modules are still used for content ingestion
- **Database**: Two database access patterns exist (SQLAlchemy in `api/`, asyncpg in `db/`)

## Recommendations

For future development:
1. Use the `api/` structure for all new API endpoints
2. Consider migrating the ingestion pipeline to use the same database approach as the main API
3. Consolidate database access to a single approach (preferably SQLAlchemy) for consistency
4. The `main.py` file represents the older API and may be deprecated in favor of the structured approach