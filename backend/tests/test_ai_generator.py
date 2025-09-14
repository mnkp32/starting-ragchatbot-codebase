"""
Unit tests for AIGenerator functionality.

Tests the AIGenerator class that handles interactions with Anthropic's Claude API
and manages tool calling for the RAG system.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator, RoundContext, ToolExecutionResult


class TestAIGenerator:
    """Test cases for AIGenerator"""
    
    def test_initialization(self):
        """Test AIGenerator initialization"""
        api_key = "test-api-key"
        model = "claude-sonnet-4-20250514"
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key, model)
            
            # Verify Anthropic client was initialized with correct API key
            mock_anthropic.assert_called_once_with(api_key=api_key)
            
            # Verify model and base parameters are set correctly
            assert generator.model == model
            assert generator.base_params["model"] == model
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test response generation without tools (direct response)"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock direct response (no tool use)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = generator.generate_response("What is machine learning?")
        
        # Verify API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == "test-model"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args
        
        # Verify response
        assert response == "Direct response without tools"

    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation includes conversation history"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock direct response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with history")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        history = "Previous conversation context"
        response = generator.generate_response("Current query", conversation_history=history)
        
        # Verify system prompt includes history
        call_args = mock_anthropic_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert "Previous conversation context" in system_content

    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client, mock_tool_manager):
        """Test response generation with tools available but no tool use"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock response that doesn't use tools
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response without using tools")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = [{"name": "search_course_content", "description": "Search course materials"}]
        response = generator.generate_response(
            "What is Python?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        # Verify API was called with tools
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # Verify response
        assert response == "Response without using tools"

    def test_generate_response_with_tool_use(self, mock_tool_manager):
        """Test complete tool execution flow"""
        generator = AIGenerator("test-key", "test-model")
        
        # Mock client with tool use response
        mock_client = Mock()
        
        # First response: tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "RAG systems"}
        mock_tool_block.id = "tool_123"
        mock_tool_response.content = [mock_tool_block]
        
        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response after tool execution")]
        
        # Set up mock to return different responses on successive calls
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        generator.client = mock_client
        
        # Mock tool manager response
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        tools = [{"name": "search_course_content"}]
        response = generator.generate_response(
            "What is RAG?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed correctly
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="RAG systems"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        # Verify final response
        assert response == "Final response after tool execution"

    def test_tool_execution_message_flow(self, mock_tool_manager):
        """Test that tool execution creates proper message flow"""
        generator = AIGenerator("test-key", "test-model")
        
        mock_client = Mock()
        
        # Mock initial tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test query"}
        mock_tool_block.id = "tool_456"
        mock_tool_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        generator.client = mock_client
        
        mock_tool_manager.execute_tool.return_value = "Tool result content"
        
        generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify the final API call has correct message structure
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = final_call_args["messages"]
        
        # Should have: [user_query, assistant_tool_use, user_tool_result]
        assert len(messages) == 3
        
        # First message: original user query
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test query"
        
        # Second message: assistant tool use
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == [mock_tool_block]
        
        # Third message: tool result
        assert messages[2]["role"] == "user"
        assert len(messages[2]["content"]) == 1
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_456"
        assert tool_result["content"] == "Tool result content"

    def test_multiple_tool_calls_in_response(self, mock_tool_manager):
        """Test handling multiple tool calls in single response"""
        generator = AIGenerator("test-key", "test-model")
        
        mock_client = Mock()
        
        # Mock response with multiple tool uses
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        # Create two tool use blocks
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "first query"}
        tool_block1.id = "tool_1"
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "search_course_content"
        tool_block2.input = {"query": "second query"}
        tool_block2.id = "tool_2"
        
        mock_tool_response.content = [tool_block1, tool_block2]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        generator.client = mock_client
        
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        response = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Check the tool results in final message
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_results_message = final_call_args["messages"][2]
        tool_results = tool_results_message["content"]
        
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[0]["content"] == "Result 1"
        assert tool_results[1]["tool_use_id"] == "tool_2"
        assert tool_results[1]["content"] == "Result 2"

    def test_system_prompt_content(self):
        """Test that system prompt contains expected instructions"""
        # Test the static system prompt
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        # Verify it contains tool usage guidelines
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        
        # Verify response protocol is included
        assert "General knowledge questions" in system_prompt
        assert "Course content questions" in system_prompt
        
        # Verify no meta-commentary instruction
        assert "no meta-commentary" in system_prompt.lower()

    def test_response_with_content_query(self, content_queries, mock_tool_manager):
        """Test that content queries should trigger tool usage"""
        generator = AIGenerator("test-key", "test-model")
        
        # This test verifies the query types that should trigger tools
        # In actual usage, Claude's decision-making would determine tool use
        # But we can test that tools are properly set up for such queries
        
        for query in content_queries:
            mock_client = Mock()
            
            # Mock tool use response for content queries
            mock_tool_response = Mock()
            mock_tool_response.stop_reason = "tool_use"
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.input = {"query": query}
            mock_tool_block.id = "tool_test"
            mock_tool_response.content = [mock_tool_block]
            
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text=f"Response for: {query}")]
            
            mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
            generator.client = mock_client
            
            mock_tool_manager.execute_tool.return_value = "Course content result"
            
            response = generator.generate_response(
                query,
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )
            
            # Verify tool was available for use
            call_args = mock_client.messages.create.call_args_list[0][1]
            assert "tools" in call_args

    def test_error_handling_in_tool_execution(self, mock_tool_manager):
        """Test error handling when tool execution fails"""
        generator = AIGenerator("test-key", "test-model")
        
        mock_client = Mock()
        
        # Mock tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_error"
        mock_tool_response.content = [mock_tool_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Error handled response")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        generator.client = mock_client
        
        # Mock tool execution failure
        mock_tool_manager.execute_tool.return_value = "Tool error: Search failed"
        
        response = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify error was passed to Claude for handling
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_result = final_call_args["messages"][2]["content"][0]
        assert tool_result["content"] == "Tool error: Search failed"
        
        # Response should still be generated
        assert response == "Error handled response"

    def test_no_tool_manager_with_tool_use(self):
        """Test behavior when tool_use occurs but no tool_manager provided"""
        generator = AIGenerator("test-key", "test-model") 
        
        mock_client = Mock()
        
        # Mock tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [Mock(text="Tool use attempt")]
        
        mock_client.messages.create.return_value = mock_tool_response
        generator.client = mock_client
        
        # Call without tool_manager
        response = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}]
            # Note: no tool_manager provided
        )
        
        # Should return the raw tool use response
        assert response == "Tool use attempt"

    # New tests for sequential tool calling functionality
    
    def test_sequential_two_round_execution(self):
        """Test successful two-round tool execution"""
        generator = AIGenerator("test-key", "test-model")
        mock_client = Mock()
        
        # Round 1: Tool use response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "get_course_outline"
        mock_tool_block1.input = {"course_name": "Python Basics"}
        mock_tool_block1.id = "tool_1"
        mock_round1_response.content = [mock_tool_block1]
        
        # Round 2: Another tool use response
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.input = {"query": "variables", "course_name": "Python Basics"}
        mock_tool_block2.id = "tool_2"
        mock_round2_response.content = [mock_tool_block2]
        
        # Round 3: Final response (max rounds reached)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final sequential response")]
        
        mock_client.messages.create.side_effect = [
            mock_round1_response, 
            mock_round2_response, 
            mock_final_response
        ]
        generator.client = mock_client
        
        # Mock tool manager responses
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline result",
            "Content search result"
        ]
        
        response = generator.generate_response(
            "Tell me about variables in Python Basics course",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify 3 API calls were made (2 tool rounds + 1 final)
        assert mock_client.messages.create.call_count == 3
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Python Basics")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="variables", course_name="Python Basics")
        
        # Verify final response
        assert response == "Final sequential response"
    
    def test_early_termination_no_tool_use(self):
        """Test early termination when first response has no tool use"""
        generator = AIGenerator("test-key", "test-model")
        mock_client = Mock()
        
        # Single response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct answer without tools")]
        mock_client.messages.create.return_value = mock_response
        generator.client = mock_client
        
        mock_tool_manager = Mock()
        
        response = generator.generate_response(
            "What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify only one API call was made
        assert mock_client.messages.create.call_count == 1
        
        # Verify no tools were executed
        assert mock_tool_manager.execute_tool.call_count == 0
        
        # Verify response
        assert response == "Direct answer without tools"
    
    def test_tool_execution_error_handling_sequential(self):
        """Test error handling during tool execution in sequential flow"""
        generator = AIGenerator("test-key", "test-model")
        mock_client = Mock()
        
        # Round 1: Tool use response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_error"
        mock_round1_response.content = [mock_tool_block]
        
        # Fallback response after error
        mock_fallback_response = Mock()
        mock_fallback_response.content = [Mock(text="Error handled gracefully")]
        
        mock_client.messages.create.side_effect = [mock_round1_response, mock_fallback_response]
        generator.client = mock_client
        
        # Mock tool execution failure
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")
        
        response = generator.generate_response(
            "Search for something",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was attempted
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify fallback response was returned
        assert response == "Error handled gracefully"
        
        # Verify two API calls were made (initial + fallback)
        assert mock_client.messages.create.call_count == 2
    
    def test_round_context_initialization(self):
        """Test RoundContext initialization"""
        tools = [{"name": "search_course_content"}]
        context = RoundContext(
            original_query="Test query",
            conversation_history="Previous conversation",
            tools=tools
        )
        
        assert context.original_query == "Test query"
        assert context.conversation_history == "Previous conversation"
        assert context.tools == tools
        assert context.current_round == 0
        assert len(context.errors) == 0
        assert len(context.messages) == 1
        assert context.messages[0]["role"] == "user"
        assert context.messages[0]["content"] == "Test query"
    
    def test_tool_execution_result_initialization(self):
        """Test ToolExecutionResult initialization"""
        result = ToolExecutionResult()
        
        assert result.failed == False
        assert result.error_message is None
        assert len(result.tool_results) == 0
        assert len(result.executed_tools) == 0
    
    def test_sequential_system_prompt_update(self):
        """Test that system prompt mentions sequential capability"""
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        # Verify sequential capability is mentioned
        assert "Up to 2 sequential tool calls allowed" in system_prompt
        assert "Sequential reasoning" in system_prompt
        
        # Verify strategic usage examples are included
        assert "get_course_outline for a course" in system_prompt
        assert "search_course_content for general topic" in system_prompt
        
        # Verify multi-step reasoning is mentioned
        assert "Multi-step reasoning" in system_prompt