"""
Test fixtures and utilities for RAG system tests.
"""
import pytest
import sys
import os
import asyncio
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
from fastapi.testclient import TestClient
import tempfile
import shutil

# Add backend directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults

@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Introduction to RAG Systems",
        course_link="https://example.com/rag-course",
        instructor="Dr. AI Expert",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is RAG?", 
                lesson_link="https://example.com/rag-course/lesson1"
            ),
            Lesson(
                lesson_number=2,
                title="Vector Embeddings",
                lesson_link="https://example.com/rag-course/lesson2"
            )
        ]
    )

@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation.",
            course_title="Introduction to RAG Systems",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Vector embeddings are numerical representations of text that capture semantic meaning.",
            course_title="Introduction to RAG Systems", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="ChromaDB is a vector database that stores and searches embeddings efficiently.",
            course_title="Introduction to RAG Systems",
            lesson_number=2,
            chunk_index=2
        )
    ]

@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation.",
            "Vector embeddings are numerical representations of text that capture semantic meaning."
        ],
        metadata=[
            {
                "course_title": "Introduction to RAG Systems",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Introduction to RAG Systems", 
                "lesson_number": 2,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.2]
    )

@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Search failed due to connection error")

@pytest.fixture
def mock_vector_store():
    """Mock vector store for unit testing"""
    mock = Mock()
    mock.search.return_value = SearchResults(
        documents=["Sample content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    mock.get_lesson_link.return_value = "https://example.com/lesson1"
    return mock

@pytest.fixture  
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generator"""
    mock_client = Mock()
    
    # Mock response without tool use - fix the text access
    mock_response_direct = Mock()
    mock_response_direct.stop_reason = "end_turn"
    mock_content = Mock()
    mock_content.text = "This is a direct response without tools"
    mock_response_direct.content = [mock_content]
    
    # Mock response with tool use
    mock_response_tool = Mock()
    mock_response_tool.stop_reason = "tool_use"
    mock_tool_use = Mock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "search_course_content"
    mock_tool_use.input = {"query": "test query"}
    mock_tool_use.id = "tool_123"
    mock_response_tool.content = [mock_tool_use]
    
    # Mock final response after tool execution - fix the text access
    mock_final_response = Mock()
    mock_final_content = Mock()
    mock_final_content.text = "This is the final response after tool execution"
    mock_final_response.content = [mock_final_content]
    
    # Default to direct response, can be overridden in individual tests
    mock_client.messages.create.return_value = mock_response_direct
    
    return mock_client

@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    mock = Mock()
    mock.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]
    mock.execute_tool.return_value = "Mock search results content"
    mock.get_last_sources.return_value = [
        {"text": "Test Course - Lesson 1", "link": "https://example.com/lesson1"}
    ]
    return mock

@pytest.fixture
def content_queries():
    """Sample content-related queries that should trigger tool usage"""
    return [
        "What is RAG?",
        "How do vector embeddings work?",
        "Explain ChromaDB functionality",
        "What are the main concepts in lesson 1?",
        "Tell me about the MCP course content"
    ]

@pytest.fixture
def outline_queries():
    """Sample outline queries that should trigger course outline tool"""
    return [
        "What lessons are in the RAG course?",
        "Show me the course outline for MCP",
        "List all lessons in the Introduction course"
    ]

@pytest.fixture
def general_queries():
    """General knowledge queries that shouldn't trigger tools"""
    return [
        "What is the weather today?",
        "How do I install Python?", 
        "What is machine learning?",
        "Hello, how are you?"
    ]

def assert_tool_called_correctly(mock_tool_manager, expected_tool_name, expected_params):
    """Utility function to verify tool was called with correct parameters"""
    mock_tool_manager.execute_tool.assert_called_with(expected_tool_name, **expected_params)

def assert_search_results_formatted(result_text: str, expected_course: str, expected_lesson: int = None):
    """Utility function to verify search results are properly formatted"""
    assert f"[{expected_course}" in result_text
    if expected_lesson:
        assert f"Lesson {expected_lesson}" in result_text

def create_mock_chroma_results(documents: List[str], metadata: List[Dict[str, Any]]) -> Dict:
    """Create mock ChromaDB query results"""
    return {
        'documents': [documents],
        'metadatas': [metadata],
        'distances': [[0.1] * len(documents)]
    }

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock = Mock()
    mock.query.return_value = (
        "This is a test response about RAG systems.",
        [{"text": "Test Course - Lesson 1", "link": "https://example.com/lesson1"}]
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to RAG", "Advanced AI Concepts"]
    }
    mock.session_manager.create_session.return_value = "test_session_123"
    mock.session_manager.clear_session.return_value = None
    mock.add_course_folder.return_value = (2, 50)
    return mock

@pytest.fixture
def test_app():
    """Create test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app with same endpoints but no static files
    app = FastAPI(title="Course Materials RAG System Test")
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models (same as main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for the app
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "Test response",
        [{"text": "Test source", "link": "https://example.com"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    mock_rag.session_manager.clear_session.return_value = None
    
    # API endpoints (same as main app)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            
            source_objects = []
            for source in sources:
                if isinstance(source, dict):
                    source_objects.append(Source(**source))
                else:
                    source_objects.append(Source(text=source, link=None))
            
            return QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        try:
            mock_rag.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "RAG System Test API"}
    
    return app

@pytest.fixture
def client(test_app):
    """Test client for API testing"""
    return TestClient(test_app)

@pytest.fixture
def temp_docs_dir():
    """Create temporary docs directory for testing"""
    temp_dir = tempfile.mkdtemp()
    docs_dir = os.path.join(temp_dir, "docs")
    os.makedirs(docs_dir)
    
    # Create sample documents
    with open(os.path.join(docs_dir, "course1.txt"), "w") as f:
        f.write("Introduction to RAG\nLesson 1: What is RAG?\nRAG combines retrieval and generation.")
    
    with open(os.path.join(docs_dir, "course2.txt"), "w") as f:
        f.write("Advanced AI\nLesson 1: Vector Embeddings\nEmbeddings represent text numerically.")
    
    yield docs_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def api_test_data():
    """Test data for API endpoint testing"""
    return {
        "query_request": {
            "query": "What is RAG?",
            "session_id": "test_session_123"
        },
        "query_request_no_session": {
            "query": "Explain vector embeddings"
        },
        "expected_response": {
            "answer": "Test response",
            "sources": [{"text": "Test source", "link": "https://example.com"}],
            "session_id": "test_session_123"
        },
        "expected_courses": {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }
    }