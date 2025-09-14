"""
Integration tests for the RAG system.

Tests the complete flow from user query to final response, including
tool registration, AI generation, and source tracking.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config


class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def rag_system(self, mock_config):
        """Create RAG system with mocked dependencies"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            system = RAGSystem(mock_config)
            
            # Mock the components
            system.vector_store = Mock()
            system.ai_generator = Mock()
            system.session_manager = Mock()
            
            return system

    def test_rag_system_initialization(self, mock_config):
        """Test RAG system initializes all components correctly"""
        with patch('rag_system.DocumentProcessor') as mock_doc_processor, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager') as mock_session_manager:
            
            system = RAGSystem(mock_config)
            
            # Verify all components were initialized with correct parameters
            mock_doc_processor.assert_called_once_with(
                mock_config.CHUNK_SIZE, 
                mock_config.CHUNK_OVERLAP
            )
            mock_vector_store.assert_called_once_with(
                mock_config.CHROMA_PATH,
                mock_config.EMBEDDING_MODEL,
                mock_config.MAX_RESULTS
            )
            mock_ai_generator.assert_called_once_with(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )
            mock_session_manager.assert_called_once_with(mock_config.MAX_HISTORY)

    def test_tool_registration(self, rag_system):
        """Test that tools are properly registered"""
        # Verify tool manager has both tools
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        
        assert len(tool_definitions) == 2
        tool_names = [tool.get("name") for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_query_processing_flow(self, rag_system, sample_search_results):
        """Test complete query processing from start to finish"""
        # Setup mocks
        rag_system.ai_generator.generate_response.return_value = "Generated response about RAG"
        rag_system.tool_manager.get_last_sources.return_value = [
            {"text": "RAG Course - Lesson 1", "link": "https://example.com/lesson1"}
        ]
        rag_system.session_manager.get_conversation_history.return_value = "Previous context"
        
        # Execute query
        response, sources = rag_system.query("What is RAG?", session_id="test_session")
        
        # Verify AI generator was called correctly
        rag_system.ai_generator.generate_response.assert_called_once()
        call_args = rag_system.ai_generator.generate_response.call_args
        
        assert "What is RAG?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] == "Previous context"
        assert call_args[1]["tools"] == rag_system.tool_manager.get_tool_definitions()
        assert call_args[1]["tool_manager"] == rag_system.tool_manager
        
        # Verify response and sources
        assert response == "Generated response about RAG"
        assert len(sources) == 1
        assert sources[0]["text"] == "RAG Course - Lesson 1"

    def test_query_without_session(self, rag_system):
        """Test query processing without session ID"""
        rag_system.ai_generator.generate_response.return_value = "Response without session"
        rag_system.tool_manager.get_last_sources.return_value = []
        
        response, sources = rag_system.query("Test query")
        
        # Verify session manager was not called for history
        rag_system.session_manager.get_conversation_history.assert_not_called()
        
        # But session manager should be called to add the exchange
        rag_system.session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator called without history
        call_args = rag_system.ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] is None

    def test_session_management(self, rag_system):
        """Test session creation and history management"""
        rag_system.ai_generator.generate_response.return_value = "Session response"
        rag_system.tool_manager.get_last_sources.return_value = []
        rag_system.session_manager.get_conversation_history.return_value = "History"
        
        response, sources = rag_system.query("Test query", session_id="session123")
        
        # Verify session history was retrieved
        rag_system.session_manager.get_conversation_history.assert_called_once_with("session123")
        
        # Verify conversation was updated
        rag_system.session_manager.add_exchange.assert_called_once_with(
            "session123", 
            "Test query", 
            "Session response"
        )

    def test_source_tracking_and_reset(self, rag_system):
        """Test that sources are properly tracked and reset"""
        rag_system.ai_generator.generate_response.return_value = "Response"
        test_sources = [
            {"text": "Course A - Lesson 1", "link": "https://example.com/a1"},
            {"text": "Course B - Lesson 2", "link": "https://example.com/b2"}
        ]
        rag_system.tool_manager.get_last_sources.return_value = test_sources
        
        response, sources = rag_system.query("Test query")
        
        # Verify sources were retrieved
        rag_system.tool_manager.get_last_sources.assert_called_once()
        assert sources == test_sources
        
        # Verify sources were reset after retrieval
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_content_query_triggers_tools(self, rag_system, content_queries):
        """Test that content-related queries have tools available"""
        rag_system.ai_generator.generate_response.return_value = "Content response"
        rag_system.tool_manager.get_last_sources.return_value = []
        
        for query in content_queries:
            rag_system.query(query)
            
            # Verify tools were provided to AI generator
            call_args = rag_system.ai_generator.generate_response.call_args[1]
            assert call_args["tools"] is not None
            assert len(call_args["tools"]) == 2  # Both search and outline tools
            assert call_args["tool_manager"] == rag_system.tool_manager

    def test_outline_query_has_tools_available(self, rag_system, outline_queries):
        """Test that outline queries have access to outline tool"""
        rag_system.ai_generator.generate_response.return_value = "Outline response"
        rag_system.tool_manager.get_last_sources.return_value = []
        
        for query in outline_queries:
            rag_system.query(query)
            
            # Verify outline tool is available
            call_args = rag_system.ai_generator.generate_response.call_args[1]
            tool_names = [tool["name"] for tool in call_args["tools"]]
            assert "get_course_outline" in tool_names

    def test_general_query_still_has_tools(self, rag_system, general_queries):
        """Test that even general queries have tools available (Claude decides usage)"""
        rag_system.ai_generator.generate_response.return_value = "General response"
        rag_system.tool_manager.get_last_sources.return_value = []
        
        for query in general_queries:
            rag_system.query(query)
            
            # Tools should still be available - Claude decides whether to use them
            call_args = rag_system.ai_generator.generate_response.call_args[1]
            assert call_args["tools"] is not None

    def test_query_prompt_formatting(self, rag_system):
        """Test that query is properly formatted as prompt"""
        rag_system.ai_generator.generate_response.return_value = "Response"
        rag_system.tool_manager.get_last_sources.return_value = []
        
        user_query = "What are vector embeddings?"
        rag_system.query(user_query)
        
        # Verify prompt includes the user query
        call_args = rag_system.ai_generator.generate_response.call_args[1]
        prompt = call_args["query"]
        assert user_query in prompt
        assert "Answer this question about course materials:" in prompt

    def test_error_handling_in_ai_generation(self, rag_system):
        """Test error handling when AI generation fails"""
        # Mock AI generator to raise exception
        rag_system.ai_generator.generate_response.side_effect = Exception("AI generation failed")
        
        with pytest.raises(Exception) as exc_info:
            rag_system.query("Test query")
        
        assert "AI generation failed" in str(exc_info.value)

    def test_course_analytics(self, rag_system):
        """Test course analytics functionality"""
        rag_system.vector_store.get_course_count.return_value = 5
        rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course A", "Course B", "Course C", "Course D", "Course E"
        ]
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]

    def test_add_course_document_flow(self, rag_system, sample_course, sample_course_chunks):
        """Test adding a single course document"""
        # Mock document processor
        rag_system.document_processor.process_course_document.return_value = (
            sample_course, sample_course_chunks
        )
        
        course, chunk_count = rag_system.add_course_document("test_file.pdf")
        
        # Verify document was processed
        rag_system.document_processor.process_course_document.assert_called_once_with("test_file.pdf")
        
        # Verify data was added to vector store
        rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_course)
        rag_system.vector_store.add_course_content.assert_called_once_with(sample_course_chunks)
        
        # Verify return values
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

    def test_add_course_document_error_handling(self, rag_system):
        """Test error handling when adding course document fails"""
        rag_system.document_processor.process_course_document.side_effect = Exception("Processing failed")
        
        course, chunk_count = rag_system.add_course_document("bad_file.pdf")
        
        # Should return None and 0 on error
        assert course is None
        assert chunk_count == 0

    def test_course_folder_processing(self, rag_system, sample_course, sample_course_chunks):
        """Test processing multiple course documents from folder"""
        # Mock file system
        with patch('os.path.exists') as mock_exists, \
             patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile:
            
            mock_exists.return_value = True
            mock_listdir.return_value = ["course1.pdf", "course2.txt", "ignore.xml"]
            mock_isfile.return_value = True
            
            # Mock document processing
            rag_system.document_processor.process_course_document.return_value = (
                sample_course, sample_course_chunks
            )
            rag_system.vector_store.get_existing_course_titles.return_value = []
            
            total_courses, total_chunks = rag_system.add_course_folder("test_folder")
            
            # Should process 2 files (PDF and TXT, but not XML)
            assert rag_system.document_processor.process_course_document.call_count == 2
            assert total_courses == 2
            assert total_chunks == len(sample_course_chunks) * 2


class TestRAGSystemWithRealComponents:
    """Integration tests using some real components (not fully mocked)"""
    
    def test_tool_manager_integration(self):
        """Test that tool manager integrates correctly with real tool objects"""
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
        from vector_store import VectorStore
        
        # Use mock vector store but real tool manager and tools
        mock_vector_store = Mock()
        
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        # Test tool definitions
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 2
        
        tool_names = [def_["name"] for def_ in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        
        # Test tool execution
        mock_vector_store.search.return_value = Mock(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
            is_empty=lambda: False
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = tool_manager.execute_tool("search_course_content", query="test")
        assert "Test Course" in result

    def test_ai_generator_with_real_tool_manager(self, mock_tool_manager):
        """Test AI generator integration with tool manager"""
        from ai_generator import AIGenerator
        
        # Mock Anthropic client
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            generator = AIGenerator("test-key", "test-model")
            
            # Mock non-tool response
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [Mock(text="Direct response")]
            mock_client.messages.create.return_value = mock_response
            
            response = generator.generate_response(
                "Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
            
            # Verify integration works
            assert response == "Direct response"
            
            # Verify tools were passed to API
            call_args = mock_client.messages.create.call_args[1]
            assert "tools" in call_args
            assert call_args["tool_choice"] == {"type": "auto"}