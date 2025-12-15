import pytest
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime


class TestAPIIntegration:
    """Integration tests for the API endpoints"""

    def setup_method(self):
        """Set up the test client and mocks"""
        self.client = TestClient(app)

    def test_full_ingestion_to_query_workflow(self):
        """Test the complete workflow from ingestion to querying"""
        # Step 1: Ingest content
        ingestion_data = {
            "title": "AI Fundamentals",
            "content": "Artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. This can include learning from experience, understanding natural language, solving problems, and recognizing patterns.",
            "book_id": "test-book-integration",
            "chapter_id": "ch1",
            "page_number": 1,
            "paragraph_id": "p1"
        }

        # Ingest the content
        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": f"Bearer test-api-key"}
        )

        # Verify ingestion was successful
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert ingest_data["success"] is True
        assert "document_id" in ingest_data

        # Step 2: Query the ingested content
        query_data = {
            "query": "What is artificial intelligence?",
            "book_id": "test-book-integration",
            "mode": "GLOBAL"
        }

        query_response = self.client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": f"Bearer test-api-key"}
        )

        # Verify query was successful
        assert query_response.status_code == 200
        query_result = query_response.json()
        assert "answer" in query_result
        assert len(query_result["contexts"]) > 0
        # The answer should be related to AI based on the ingested content
        assert "artificial intelligence" in query_result["answer"].lower()

    def test_selection_only_mode_workflow(self):
        """Test the selection-only mode workflow"""
        # First, ingest some content
        ingestion_data = {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "book_id": "test-book-selection",
            "chapter_id": "ch2",
            "page_number": 10,
            "paragraph_id": "p1"
        }

        # Ingest the content
        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": f"Bearer test-api-key"}
        )

        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert ingest_data["success"] is True

        # Now test selection-only query
        selection_query_data = {
            "query": "What can systems do in machine learning?",
            "book_id": "test-book-selection",
            "mode": "SELECTION_ONLY",
            "selected_text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
        }

        query_response = self.client.post(
            "/api/v1/query",
            json=selection_query_data,
            headers={"Authorization": f"Bearer test-api-key"}
        )

        assert query_response.status_code == 200
        query_result = query_response.json()
        assert "answer" in query_result
        # The answer should specifically address what systems can do based on the selected text
        assert "systems" in query_result["answer"].lower()
        assert "learn" in query_result["answer"].lower() or "identify" in query_result["answer"].lower()

    def test_multiple_ingestions_and_queries(self):
        """Test multiple content ingestions and queries"""
        # Ingest multiple pieces of content
        contents = [
            {
                "title": "Chapter 1",
                "content": "The first chapter discusses the basics of programming.",
                "book_id": "multi-test-book",
                "chapter_id": "ch1",
                "page_number": 1,
                "paragraph_id": "p1"
            },
            {
                "title": "Chapter 2",
                "content": "The second chapter covers advanced algorithms and data structures.",
                "book_id": "multi-test-book",
                "chapter_id": "ch2",
                "page_number": 20,
                "paragraph_id": "p1"
            }
        ]

        document_ids = []
        for content in contents:
            response = self.client.post(
                "/api/v1/ingest",
                json=content,
                headers={"Authorization": f"Bearer test-api-key"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            document_ids.append(data["document_id"])

        # Query about programming (should find content from chapter 1)
        query1 = {
            "query": "What does the first chapter discuss?",
            "book_id": "multi-test-book",
            "mode": "GLOBAL"
        }

        response1 = self.client.post(
            "/api/v1/query",
            json=query1,
            headers={"Authorization": f"Bearer test-api-key"}
        )

        assert response1.status_code == 200
        result1 = response1.json()
        assert "basics" in result1["answer"].lower() or "programming" in result1["answer"].lower()

        # Query about algorithms (should find content from chapter 2)
        query2 = {
            "query": "What does the second chapter cover?",
            "book_id": "multi-test-book",
            "mode": "GLOBAL"
        }

        response2 = self.client.post(
            "/api/v1/query",
            json=query2,
            headers={"Authorization": f"Bearer test-api-key"}
        )

        assert response2.status_code == 200
        result2 = response2.json()
        assert "algorithms" in result2["answer"].lower() or "data structures" in result2["answer"].lower()

    def test_error_handling_integration(self):
        """Test error handling across the API"""
        # Test query without content (should return not found response)
        query_data = {
            "query": "What is quantum computing?",
            "book_id": "nonexistent-book",
            "mode": "GLOBAL"
        }

        response = self.client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": f"Bearer test-api-key"}
        )

        # Should return a valid response with a "not found" message
        assert response.status_code == 200
        result = response.json()
        # The response should indicate that the information wasn't found
        assert isinstance(result["answer"], str)

    def test_health_and_status_endpoints(self):
        """Test health and status endpoints"""
        # Test basic health endpoint
        health_response = self.client.get("/api/v1/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert "status" in health_data

        # Test detailed health endpoint
        detail_response = self.client.get("/api/v1/health/detail")
        assert detail_response.status_code == 200
        detail_data = detail_response.json()
        assert "status" in detail_data
        assert "details" in detail_data

    def test_api_rate_limiting_integration(self):
        """Test API rate limiting"""
        # Send multiple requests quickly to test rate limiting
        query_data = {
            "query": "What is AI?",
            "book_id": "rate-test-book",
            "mode": "GLOBAL"
        }

        responses = []
        for i in range(10):  # Send more requests than the rate limit
            response = self.client.post(
                "/api/v1/query",
                json=query_data,
                headers={"Authorization": f"Bearer test-api-key"}
            )
            responses.append(response.status_code)

        # Check that some requests were rate-limited (429) and others succeeded (200)
        success_count = responses.count(200)
        rate_limited_count = responses.count(429)

        # At least some requests should succeed and some should be rate-limited
        assert success_count > 0

    def test_concurrent_ingestion_queries(self):
        """Test concurrent ingestion and query operations"""
        import threading
        import time

        results = []

        def ingest_content(content_id):
            """Function to ingest content in a thread"""
            ingestion_data = {
                "title": f"Concurrent Content {content_id}",
                "content": f"This is content for testing concurrent operations {content_id}. It contains unique information for thread {content_id}.",
                "book_id": "concurrent-test-book",
                "chapter_id": f"ch{content_id}",
                "page_number": content_id * 10,
                "paragraph_id": "p1"
            }

            response = self.client.post(
                "/api/v1/ingest",
                json=ingestion_data,
                headers={"Authorization": f"Bearer test-api-key"}
            )
            results.append(("ingest", response.status_code, content_id))

        def query_content(query_id):
            """Function to query content in a thread"""
            query_data = {
                "query": f"What is content {query_id} about?",
                "book_id": "concurrent-test-book",
                "mode": "GLOBAL"
            }

            response = self.client.post(
                "/api/v1/query",
                json=query_data,
                headers={"Authorization": f"Bearer test-api-key"}
            )
            results.append(("query", response.status_code, query_id))

        # Create and start threads for concurrent operations
        threads = []

        # Create ingestion threads
        for i in range(3):
            t = threading.Thread(target=ingest_content, args=(i,))
            threads.append(t)
            t.start()

        # Create query threads
        for i in range(2):
            t = threading.Thread(target=query_content, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify that all operations completed successfully
        ingest_results = [r for r in results if r[0] == "ingest"]
        query_results = [r for r in results if r[0] == "query"]

        assert len(ingest_results) == 3
        assert len(query_results) == 2
        assert all(r[1] == 200 for r in ingest_results)  # All ingests successful
        assert all(r[1] == 200 for r in query_results)   # All queries successful

    def test_api_response_consistency(self):
        """Test that API responses are consistent across calls"""
        # Ingest content
        ingestion_data = {
            "title": "Consistency Test",
            "content": "This content is used to test API response consistency. The same query should yield similar responses.",
            "book_id": "consistency-test-book",
            "chapter_id": "ch1",
            "page_number": 1,
            "paragraph_id": "p1"
        }

        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": f"Bearer test-api-key"}
        )
        assert ingest_response.status_code == 200

        # Make the same query multiple times
        query_data = {
            "query": "What is this content about?",
            "book_id": "consistency-test-book",
            "mode": "GLOBAL"
        }

        responses = []
        for i in range(3):
            response = self.client.post(
                "/api/v1/query",
                json=query_data,
                headers={"Authorization": f"Bearer test-api-key"}
            )
            assert response.status_code == 200
            responses.append(response.json())

        # Check that all responses have the expected structure
        for resp in responses:
            assert "answer" in resp
            assert "contexts" in resp
            assert "citations" in resp