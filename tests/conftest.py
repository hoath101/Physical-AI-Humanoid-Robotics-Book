import pytest
from unittest.mock import Mock, patch
from api.main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the API"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_external_services():
    """Automatically mock external services for all tests"""
    with patch('api.services.embedding.openai_client'), \
         patch('api.config.vector_db.QdrantClient'), \
         patch('api.config.database.create_engine'), \
         patch('api.middleware.rate_limiter.limiter'), \
         patch('api.utils.logger.logger'):
        yield


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    with patch('api.config.settings.settings') as mock:
        mock.OPENAI_API_KEY = "test-openai-key"
        mock.QDRANT_URL = "http://test-qdrant:6333"
        mock.QDRANT_API_KEY = "test-qdrant-key"
        mock.NEON_DATABASE_URL = "postgresql://test:test@test:5432/test"
        mock.API_KEY = "test-api-key"
        mock.EMBEDDING_DIMENSION = 1536
        mock.CHUNK_SIZE = 1000
        mock.CHUNK_OVERLAP = 200
        mock.TOP_K = 5
        mock.MIN_SCORE = 0.5
        mock.RATE_LIMIT_REQUESTS = 100
        mock.RATE_LIMIT_WINDOW = 60
        yield mock


@pytest.fixture
def sample_book_content():
    """Sample book content for testing"""
    return {
        "title": "Test Chapter 1",
        "content": "This is sample content for testing purposes. It contains multiple sentences to test various functionality.",
        "book_id": "test-book-1",
        "chapter_id": "ch1",
        "page_number": 1,
        "paragraph_id": "p1"
    }


@pytest.fixture
def sample_query_request():
    """Sample query request for testing"""
    return {
        "query": "What is this book about?",
        "book_id": "test-book-1",
        "mode": "GLOBAL"
    }


@pytest.fixture
def sample_selection_query_request():
    """Sample selection query request for testing"""
    return {
        "query": "Explain this concept?",
        "book_id": "test-book-1",
        "mode": "SELECTION_ONLY",
        "selected_text": "This is the selected text for testing."
    }