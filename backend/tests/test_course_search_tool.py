"""
Unit tests for CourseSearchTool functionality.

Tests the CourseSearchTool class that provides search capabilities for course content
through the Anthropic tool calling interface.
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""
    
    def test_tool_definition_structure(self, mock_vector_store):
        """Test that tool definition has correct structure for Anthropic API"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        # Verify required fields
        assert "name" in definition
        assert "description" in definition  
        assert "input_schema" in definition
        
        # Verify tool name
        assert definition["name"] == "search_course_content"
        
        # Verify schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        
        # Verify required parameters
        assert "query" in schema["required"]
        
        # Verify optional parameters exist
        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
        
        # Verify parameter types
        assert properties["query"]["type"] == "string"
        assert properties["course_name"]["type"] == "string"
        assert properties["lesson_number"]["type"] == "integer"

    def test_execute_basic_search(self, mock_vector_store, sample_search_results):
        """Test basic search execution without filters"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("What is RAG?")
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="What is RAG?",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result contains expected content
        assert "[Introduction to RAG Systems - Lesson 1]" in result
        assert "[Introduction to RAG Systems - Lesson 2]" in result
        assert "RAG stands for Retrieval-Augmented Generation" in result
        assert "Vector embeddings are numerical representations" in result
        
        # Verify sources were tracked
        assert len(tool.last_sources) > 0

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test search execution with course name filter"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("What is RAG?", course_name="RAG Systems")
        
        # Verify vector store was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="What is RAG?",
            course_name="RAG Systems",
            lesson_number=None
        )
        
        assert "Introduction to RAG Systems" in result

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search execution with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("What is RAG?", lesson_number=1)
        
        # Verify vector store was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="What is RAG?",
            course_name=None,
            lesson_number=1
        )
        
        assert "Lesson 1" in result

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test search execution with both course and lesson filters"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("What is RAG?", course_name="RAG Systems", lesson_number=1)
        
        # Verify vector store was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="What is RAG?",
            course_name="RAG Systems", 
            lesson_number=1
        )

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("non-existent content")
        
        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, mock_vector_store, empty_search_results):
        """Test empty results message includes filter information"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Non-existent Course", lesson_number=99)
        
        assert "No relevant content found in course 'Non-existent Course' in lesson 99" in result

    def test_execute_error_results(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "Search failed due to connection error" == result

    def test_result_formatting(self, mock_vector_store):
        """Test proper formatting of search results"""
        # Create specific test data for formatting
        test_results = SearchResults(
            documents=["First content chunk", "Second content chunk"],
            metadata=[
                {"course_title": "Test Course A", "lesson_number": 1},
                {"course_title": "Test Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = test_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/courseA/lesson1",
            "https://example.com/courseB/lesson2"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify formatting structure
        assert "[Test Course A - Lesson 1]" in result
        assert "[Test Course B - Lesson 2]" in result
        assert "First content chunk" in result
        assert "Second content chunk" in result
        
        # Verify chunks are separated properly
        chunks = result.split("\n\n")
        assert len(chunks) == 2

    def test_source_tracking(self, mock_vector_store):
        """Test that sources are properly tracked for UI display"""
        test_results = SearchResults(
            documents=["Content from course A", "Content from course B"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = test_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/courseA/lesson1",
            "https://example.com/courseB/lesson2"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        tool.execute("test query")
        
        # Verify sources were tracked
        assert len(tool.last_sources) == 2
        
        # Verify source structure
        source1 = tool.last_sources[0]
        source2 = tool.last_sources[1]
        
        assert source1["text"] == "Course A - Lesson 1"
        assert source1["link"] == "https://example.com/courseA/lesson1"
        assert source2["text"] == "Course B - Lesson 2" 
        assert source2["link"] == "https://example.com/courseB/lesson2"

    def test_source_deduplication(self, mock_vector_store):
        """Test that duplicate sources are properly deduplicated"""
        # Create results with duplicate course+lesson combinations
        test_results = SearchResults(
            documents=["First chunk from lesson 1", "Second chunk from lesson 1", "Chunk from lesson 2"],
            metadata=[
                {"course_title": "Same Course", "lesson_number": 1},
                {"course_title": "Same Course", "lesson_number": 1},  # Duplicate
                {"course_title": "Same Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2, 0.3]
        )
        mock_vector_store.search.return_value = test_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        
        tool = CourseSearchTool(mock_vector_store)
        tool.execute("test query")
        
        # Should have only 2 unique sources despite 3 chunks
        assert len(tool.last_sources) == 2
        
        source_texts = [source["text"] for source in tool.last_sources]
        assert "Same Course - Lesson 1" in source_texts
        assert "Same Course - Lesson 2" in source_texts

    def test_source_without_lesson_number(self, mock_vector_store):
        """Test source formatting when lesson_number is None"""
        test_results = SearchResults(
            documents=["Content without lesson number"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": None}
            ],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = test_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should format without lesson number
        assert "[Test Course]" in result
        assert "Lesson" not in result.split("[Test Course]")[1].split("]")[0]
        
        # Source should not include lesson number
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None


class TestToolManager:
    """Test cases for ToolManager"""
    
    def test_tool_registration(self, mock_vector_store):
        """Test tool registration in ToolManager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting tool definitions for Anthropic API"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test tool execution through ToolManager"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "Introduction to RAG Systems" in result

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test execution of non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last search operation"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test query")
        
        sources = manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources after retrieval"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0