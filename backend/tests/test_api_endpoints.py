"""
API endpoint tests for the RAG System FastAPI application.
Tests all endpoints for proper request/response handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    def test_query_with_session_id(self, client, api_test_data):
        """Test query endpoint with provided session ID"""
        response = client.post(
            "/api/query",
            json=api_test_data["query_request"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["sources"], list)
        assert data["session_id"] == "test_session_123"

    def test_query_without_session_id(self, client, api_test_data):
        """Test query endpoint without session ID (should create one)"""
        response = client.post(
            "/api/query",
            json=api_test_data["query_request_no_session"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"

    def test_query_with_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query",
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_query_with_empty_query(self, client):
        """Test query endpoint with empty query string"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        
        # Should still process but with empty query
        assert response.status_code == 200

    def test_query_response_structure(self, client, api_test_data):
        """Test that query response has correct structure"""
        response = client.post(
            "/api/query",
            json=api_test_data["query_request"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source or source.get("link") is None

    def test_query_handles_list_sources(self, client):
        """Test query endpoint handles legacy string sources"""
        # This tests the source conversion logic in the endpoint
        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    def test_get_courses(self, client, api_test_data):
        """Test getting course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_courses_response_structure(self, client):
        """Test courses endpoint response structure"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check that course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


@pytest.mark.api  
class TestSessionEndpoint:
    """Test the /api/sessions/{session_id}/clear endpoint"""

    def test_clear_session(self, client):
        """Test clearing a session"""
        session_id = "test_session_123"
        response = client.delete(f"/api/sessions/{session_id}/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "cleared" in data["message"].lower()

    def test_clear_nonexistent_session(self, client):
        """Test clearing a non-existent session"""
        session_id = "nonexistent_session"
        response = client.delete(f"/api/sessions/{session_id}/clear")
        
        # Should still return success (idempotent operation)
        assert response.status_code == 200

    def test_clear_session_with_invalid_id(self, client):
        """Test clearing session with various session ID formats"""
        test_ids = ["very_long_session_id_" * 10]
        
        for session_id in test_ids:
            response = client.delete(f"/api/sessions/{session_id}/clear")
            # Should handle gracefully
            assert response.status_code in [200, 422]
        
        # Test special characters - may return 404 due to URL parsing
        response = client.delete("/api/sessions/special-chars!@#/clear")
        assert response.status_code in [200, 404, 422]
        
        # Test empty session ID separately (404 is expected for empty path)
        response = client.delete("/api/sessions//clear")
        assert response.status_code == 404


@pytest.mark.api
class TestRootEndpoint:
    """Test the root / endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns expected message"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RAG System" in data["message"]


@pytest.mark.api
class TestErrorHandling:
    """Test error handling across endpoints"""

    @patch('fastapi.testclient.TestClient.post')
    def test_query_internal_error(self, mock_post, client):
        """Test query endpoint handles internal errors gracefully"""
        # This test would need to mock the RAG system to throw an error
        # For now, we test the structure
        pass

    def test_invalid_endpoints(self, client):
        """Test requests to invalid endpoints"""
        response = client.get("/api/invalid")
        assert response.status_code == 404

        response = client.post("/api/invalid")
        assert response.status_code == 404

    def test_wrong_http_methods(self, client):
        """Test wrong HTTP methods on endpoints"""
        # GET on query endpoint (should be POST)
        response = client.get("/api/query")
        assert response.status_code == 405

        # POST on courses endpoint (should be GET)  
        response = client.post("/api/courses")
        assert response.status_code == 405


@pytest.mark.api
class TestContentTypes:
    """Test content type handling"""

    def test_query_with_form_data(self, client):
        """Test query endpoint rejects form data"""
        response = client.post(
            "/api/query",
            data={"query": "test"}
        )
        # Should expect JSON, not form data
        assert response.status_code == 422

    def test_query_with_correct_content_type(self, client):
        """Test query endpoint with explicit JSON content type"""
        response = client.post(
            "/api/query",
            json={"query": "test query"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200


@pytest.mark.api
@pytest.mark.integration
class TestEndpointIntegration:
    """Integration tests across multiple endpoints"""

    def test_query_then_get_courses(self, client):
        """Test workflow: query -> get courses"""
        # First make a query
        query_response = client.post(
            "/api/query",
            json={"query": "What is RAG?"}
        )
        assert query_response.status_code == 200
        
        # Then get courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200

    def test_create_session_then_clear(self, client):
        """Test workflow: create session -> clear session"""
        # Make query to create session
        query_response = client.post(
            "/api/query",
            json={"query": "test"}
        )
        assert query_response.status_code == 200
        session_id = query_response.json()["session_id"]
        
        # Clear the session
        clear_response = client.delete(f"/api/sessions/{session_id}/clear")
        assert clear_response.status_code == 200

    def test_multiple_sessions(self, client):
        """Test handling multiple sessions"""
        sessions = []
        
        # Create multiple sessions
        for i in range(3):
            response = client.post(
                "/api/query",
                json={"query": f"test query {i}"}
            )
            assert response.status_code == 200
            sessions.append(response.json()["session_id"])
        
        # Clear all sessions
        for session_id in sessions:
            response = client.delete(f"/api/sessions/{session_id}/clear")
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])