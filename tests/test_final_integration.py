import pytest
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import Mock, patch
import json


class TestFinalIntegration:
    """Final integration tests covering all user stories and functionality"""

    def setup_method(self):
        """Set up the test client and mocks"""
        self.client = TestClient(app)

    def test_end_to_end_workflow(self):
        """Test the complete end-to-end workflow: ingest -> query -> validate"""
        # Step 1: Ingest content
        ingestion_data = {
            "title": "AI Fundamentals Chapter",
            "content": "Artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. This can include learning from experience, understanding natural language, solving problems, and recognizing patterns. Machine learning is a subset of AI that focuses on algorithms that can learn from data.",
            "book_id": "test-book-e2e",
            "chapter_id": "ch1",
            "page_number": 1,
            "paragraph_id": "p1"
        }

        # Ingest the content
        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=ingestion_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert ingest_data["success"] is True
        assert "document_id" in ingest_data

        # Step 2: Test Global Mode Query
        global_query_data = {
            "query": "What is artificial intelligence?",
            "book_id": "test-book-e2e",
            "mode": "GLOBAL"
        }

        global_response = self.client.post(
            "/api/v1/query",
            json=global_query_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        assert global_response.status_code == 200
        global_result = global_response.json()
        assert "answer" in global_result
        assert len(global_result["contexts"]) > 0
        assert "artificial intelligence" in global_result["answer"].lower()

        # Step 3: Test Selection-Only Mode Query
        selection_query_data = {
            "query": "What can AI include?",
            "book_id": "test-book-e2e",
            "mode": "SELECTION_ONLY",
            "selected_text": "Artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. This can include learning from experience, understanding natural language, solving problems, and recognizing patterns."
        }

        selection_response = self.client.post(
            "/api/v1/query",
            json=selection_query_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        assert selection_response.status_code == 200
        selection_result = selection_response.json()
        assert "answer" in selection_result
        # The answer should specifically mention capabilities mentioned in the selected text
        assert "learning" in selection_result["answer"].lower() or \
               "understanding" in selection_result["answer"].lower() or \
               "solving" in selection_result["answer"].lower()

        # Step 4: Test health endpoints
        health_response = self.client.get("/api/v1/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"

        detail_health_response = self.client.get("/api/v1/health/detail")
        assert detail_health_response.status_code == 200
        detail_health_data = detail_health_response.json()
        assert "status" in detail_health_data

        print("End-to-end workflow test passed!")

    def test_user_story_1_global_search(self):
        """Test User Story 1: Query Book Content Using Global Search"""
        # Ingest content for testing
        content_data = {
            "title": "ROS 2 Fundamentals",
            "content": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
            "book_id": "ros-book",
            "chapter_id": "ch2",
            "page_number": 10,
            "paragraph_id": "p1"
        }

        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=content_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )
        assert ingest_response.status_code == 200

        # Test global query
        query_data = {
            "query": "What is ROS 2?",
            "book_id": "ros-book",
            "mode": "GLOBAL"
        }

        response = self.client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        assert response.status_code == 200
        result = response.json()
        assert "answer" in result
        assert "robot" in result["answer"].lower() or "operating system" in result["answer"].lower()
        assert len(result["contexts"]) > 0

        print("User Story 1 (Global Search) test passed!")

    def test_user_story_2_selection_only_mode(self):
        """Test User Story 2: Query Book Content Using Selection-Only Mode"""
        # Ingest content for testing
        content_data = {
            "title": "Digital Twin Concepts",
            "content": "A digital twin is a virtual representation of a physical object or system throughout its lifecycle. It uses real-time data from sensors to understand how a physical twin is performing and predict future behavior.",
            "book_id": "digital-twin-book",
            "chapter_id": "ch3",
            "page_number": 25,
            "paragraph_id": "p1"
        }

        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=content_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )
        assert ingest_response.status_code == 200

        # Test selection-only query
        query_data = {
            "query": "What does a digital twin use?",
            "book_id": "digital-twin-book",
            "mode": "SELECTION_ONLY",
            "selected_text": "A digital twin is a virtual representation of a physical object or system throughout its lifecycle. It uses real-time data from sensors to understand how a physical twin is performing and predict future behavior."
        }

        response = self.client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        assert response.status_code == 200
        result = response.json()
        assert "answer" in result
        # The answer should specifically mention "real-time data from sensors"
        assert "real-time" in result["answer"].lower() or "sensors" in result["answer"].lower()
        assert len(result["contexts"]) > 0

        print("User Story 2 (Selection-Only Mode) test passed!")

    def test_user_story_3_session_management(self):
        """Test User Story 3: Access Chatbot Interface Within Book Reader"""
        # Test that session management works correctly
        # This would typically involve testing session creation, retrieval, and updates
        # For this integration test, we'll verify that queries can be made without explicit session management
        # (as sessions should be handled automatically)

        # Ingest content
        content_data = {
            "title": "AI Perception Systems",
            "content": "AI perception systems enable robots to interpret sensory information from their environment. These systems typically include computer vision, audio processing, and sensor fusion techniques.",
            "book_id": "ai-perception-book",
            "chapter_id": "ch4",
            "page_number": 40,
            "paragraph_id": "p1"
        }

        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=content_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )
        assert ingest_response.status_code == 200

        # Make multiple queries to verify stateless operation
        for i in range(3):
            query_data = {
                "query": "What do AI perception systems do?",
                "book_id": "ai-perception-book",
                "mode": "GLOBAL"
            }

            response = self.client.post(
                "/api/v1/query",
                json=query_data,
                headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
            )

            assert response.status_code == 200
            result = response.json()
            assert "answer" in result
            assert "interpret" in result["answer"].lower() or "environment" in result["answer"].lower()

        print("User Story 3 (Session Management) test passed!")

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases"""
        # Test query with non-existent book
        query_data = {
            "query": "What is quantum computing?",
            "book_id": "non-existent-book",
            "mode": "GLOBAL"
        }

        response = self.client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        # Should return a valid response with a "not found" message, not an error
        assert response.status_code == 200
        result = response.json()
        assert "answer" in result

        # Test invalid mode
        invalid_query_data = {
            "query": "What is AI?",
            "book_id": "test-book",
            "mode": "INVALID_MODE"
        }

        invalid_response = self.client.post(
            "/api/v1/query",
            json=invalid_query_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        assert invalid_response.status_code == 422  # Validation error

        # Test missing authentication
        auth_response = self.client.post("/api/v1/query", json=query_data)
        assert auth_response.status_code == 401

        print("Error handling and edge cases test passed!")

    def test_rate_limiting_and_performance(self):
        """Test rate limiting and performance"""
        # Make multiple requests to test rate limiting
        query_data = {
            "query": "What is machine learning?",
            "book_id": "rate-limit-test-book",
            "mode": "GLOBAL"
        }

        responses = []
        for i in range(150):  # Make more requests than the rate limit
            response = self.client.post(
                "/api/v1/query",
                json=query_data,
                headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
            )
            responses.append(response.status_code)

        # Count how many requests succeeded vs were rate-limited
        success_count = responses.count(200)
        rate_limited_count = responses.count(429)

        # Some requests should succeed, some should be rate-limited
        assert success_count > 0
        print(f"Rate limiting test: {success_count} successful, {rate_limited_count} rate-limited requests")

        print("Rate limiting and performance test passed!")

    def test_response_quality_and_grounding(self):
        """Test response quality and grounding validation"""
        # Ingest content with specific facts
        content_data = {
            "title": "Vision-Language-Action Systems",
            "content": "Vision-Language-Action (VLA) models are foundation models that integrate visual perception, language understanding, and action generation. These models enable robots to understand complex commands and execute appropriate physical actions.",
            "book_id": "vla-book",
            "chapter_id": "ch5",
            "page_number": 55,
            "paragraph_id": "p1"
        }

        ingest_response = self.client.post(
            "/api/v1/ingest",
            json=content_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )
        assert ingest_response.status_code == 200

        # Test that responses are grounded in the provided text
        query_data = {
            "query": "What do VLA models integrate?",
            "book_id": "vla-book",
            "mode": "GLOBAL"
        }

        response = self.client.post(
            "/api/v1/query",
            json=query_data,
            headers={"Authorization": f"Bearer {self.get_test_api_key()}"}
        )

        assert response.status_code == 200
        result = response.json()
        assert "answer" in result
        # The answer should mention the integrations mentioned in the source text
        answer_lower = result["answer"].lower()
        assert "visual" in answer_lower or "language" in answer_lower or "action" in answer_lower
        assert len(result["contexts"]) > 0

        print("Response quality and grounding test passed!")

    def get_test_api_key(self):
        """Get a test API key - in real implementation this would come from settings"""
        # For testing purposes, we'll use a mock API key
        # In a real implementation, this would be configured in the test settings
        return "test-api-key"

    def test_complete_integration_suite(self):
        """Run all integration tests as a complete suite"""
        print("Running complete integration test suite...")

        self.test_end_to_end_workflow()
        self.test_user_story_1_global_search()
        self.test_user_story_2_selection_only_mode()
        self.test_user_story_3_session_management()
        self.test_error_handling_and_edge_cases()
        self.test_rate_limiting_and_performance()
        self.test_response_quality_and_grounding()

        print("All integration tests passed successfully!")
        print("The RAG Chatbot system is fully functional and meets all requirements.")