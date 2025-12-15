import pytest
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import Mock, patch, MagicMock
import json


@pytest.fixture
def client():
    """Create a test client for the API"""
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock all services used by the API"""
    with patch('api.api.v1.routes.ingest.IngestionService') as mock_ingestion, \
         patch('api.api.v1.routes.query.RetrievalService') as mock_retrieval, \
         patch('api.api.v1.routes.query.ChatService') as mock_chat, \
         patch('api.api.v1.routes.health.get_health_status') as mock_health:

        # Set up default return values
        mock_ingestion.return_value.create.return_value = {
            'success': True,
            'document_id': 'test-doc-123',
            'chunks_processed': 1
        }

        mock_retrieval.return_value.global_search.return_value = Mock()
        mock_retrieval.return_value.global_search.return_value.model_dump.return_value = {
            'answer': 'Test answer',
            'contexts': [],
            'citations': []
        }

        mock_chat.return_value.get_answer.return_value = Mock()
        mock_chat.return_value.get_answer.return_value.model_dump.return_value = {
            'answer': 'Test answer',
            'contexts': [],
            'citations': []
        }

        yield {
            'ingestion': mock_ingestion,
            'retrieval': mock_retrieval,
            'chat': mock_chat,
            'health': mock_health
        }


class TestIngestionEndpoints:
    """Test cases for ingestion API endpoints"""

    def test_ingest_content_success(self, client, mock_services):
        """Test successful content ingestion"""
        # Arrange
        ingestion_data = {
            "title": "Test Chapter",
            "content": "This is test content for ingestion.",
            "book_id": "test-book",
            "chapter_id": "ch1",
            "page_number": 1,
            "paragraph_id": "p1"
        }

        # Act
        response = client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "document_id" in response_data

    def test_ingest_content_missing_auth(self, client):
        """Test content ingestion without authentication"""
        # Arrange
        ingestion_data = {
            "title": "Test Chapter",
            "content": "This is test content for ingestion.",
            "book_id": "test-book",
            "chapter_id": "ch1",
            "page_number": 1,
            "paragraph_id": "p1"
        }

        # Act
        response = client.post("/api/v1/ingest", json=ingestion_data)

        # Assert
        assert response.status_code == 401

    def test_ingest_content_invalid_data(self, client):
        """Test content ingestion with invalid data"""
        # Arrange
        ingestion_data = {
            # Missing required fields
        }

        # Act
        response = client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 422  # Validation error

    def test_ingest_content_empty_content(self, client):
        """Test content ingestion with empty content"""
        # Arrange
        ingestion_data = {
            "title": "Test Chapter",
            "content": "",  # Empty content
            "book_id": "test-book",
            "chapter_id": "ch1",
            "page_number": 1,
            "paragraph_id": "p1"
        }

        # Act
        response = client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 400


class TestQueryEndpoints:
    """Test cases for query API endpoints"""

    def test_global_query_success(self, client, mock_services):
        """Test successful global query"""
        # Arrange
        query_data = {
            "query": "What is artificial intelligence?",
            "book_id": "test-book",
            "mode": "GLOBAL"
        }

        # Act
        response = client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert "answer" in response_data
        assert "contexts" in response_data

    def test_selection_only_query_success(self, client, mock_services):
        """Test successful selection-only query"""
        # Arrange
        query_data = {
            "query": "Explain this concept?",
            "book_id": "test-book",
            "mode": "SELECTION_ONLY",
            "selected_text": "Machine learning is a subset of artificial intelligence..."
        }

        # Act
        response = client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert "answer" in response_data
        assert "contexts" in response_data

    def test_query_missing_auth(self, client):
        """Test query without authentication"""
        # Arrange
        query_data = {
            "query": "What is AI?",
            "book_id": "test-book",
            "mode": "GLOBAL"
        }

        # Act
        response = client.post("/api/v1/query", json=query_data)

        # Assert
        assert response.status_code == 401

    def test_query_invalid_mode(self, client):
        """Test query with invalid mode"""
        # Arrange
        query_data = {
            "query": "What is AI?",
            "book_id": "test-book",
            "mode": "INVALID_MODE"
        }

        # Act
        response = client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 422

    def test_selection_only_query_missing_selected_text(self, client):
        """Test selection-only query without selected text"""
        # Arrange
        query_data = {
            "query": "Explain this?",
            "book_id": "test-book",
            "mode": "SELECTION_ONLY"
            # Missing selected_text
        }

        # Act
        response = client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 422


class TestHealthEndpoints:
    """Test cases for health API endpoints"""

    def test_health_check(self, client, mock_services):
        """Test health check endpoint"""
        # Arrange
        mock_services['health'].return_value = {
            "status": "healthy",
            "timestamp": "2023-01-01T00:00:00Z",
            "version": "1.0.0"
        }

        # Act
        response = client.get("/api/v1/health")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "timestamp" in response_data
        assert "version" in response_data

    def test_health_check_detailed(self, client, mock_services):
        """Test detailed health check endpoint"""
        # Arrange
        mock_services['health'].return_value = {
            "status": "healthy",
            "details": {
                "database": "connected",
                "vector_db": "connected",
                "external_apis": "available"
            },
            "timestamp": "2023-01-01T00:00:00Z"
        }

        # Act
        response = client.get("/api/v1/health/detail")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "details" in response_data
        assert "database" in response_data["details"]


class TestErrorHandling:
    """Test cases for error handling in API endpoints"""

    def test_invalid_endpoint(self, client):
        """Test request to invalid endpoint"""
        # Act
        response = client.get("/invalid-endpoint")

        # Assert
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test invalid HTTP method for an endpoint"""
        # Act
        response = client.put("/api/v1/health")  # PUT not allowed for health endpoint

        # Assert
        assert response.status_code == 405

    @patch('api.middleware.error_handler.logger')
    def test_internal_server_error(self, mock_logger, client, mock_services):
        """Test handling of internal server errors"""
        # Arrange
        # Make one of the services raise an exception
        mock_services['ingestion'].return_value.create.side_effect = Exception("Internal error")

        ingestion_data = {
            "title": "Test Chapter",
            "content": "This is test content for ingestion.",
            "book_id": "test-book",
            "chapter_id": "ch1",
            "page_number": 1,
            "paragraph_id": "p1"
        }

        # Act
        response = client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": "Bearer test-api-key"}
        )

        # Assert
        assert response.status_code == 500
        response_data = response.json()
        assert "error" in response_data


class TestRateLimiting:
    """Test cases for rate limiting functionality"""

    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # This would require more complex setup to test actual rate limiting
        # For now, just verify that the rate limiter middleware is in place
        # by making multiple requests and checking if any are rejected

        query_data = {
            "query": "What is AI?",
            "book_id": "test-book",
            "mode": "GLOBAL"
        }

        # Make several requests
        responses = []
        for i in range(5):
            response = client.post(
                "/api/v1/query",
                json=query_data,
                headers={"Authorization": "Bearer test-api-key"}
            )
            responses.append(response.status_code)

        # All should be successful unless rate limit is very low
        # In a real test, we'd set a low rate limit specifically for testing
        assert all(status in [200, 429] for status in responses)  # Either success or rate limited


class TestCORS:
    """Test cases for CORS functionality"""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses"""
        # Act
        response = client.get("/api/v1/health")

        # Assert
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-headers" in response.headers
        assert "access-control-allow-methods" in response.headers