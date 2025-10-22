from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic


@dataclass
class RoundContext:
    """Manages state and context across sequential tool calling rounds"""

    original_query: str
    conversation_history: Optional[str]
    tools: List[Dict[str, Any]]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize messages with the original query"""
        if not self.messages:
            self.messages = [{"role": "user", "content": self.original_query}]


@dataclass
class ToolExecutionResult:
    """Result of executing tools in a round"""

    failed: bool = False
    error_message: Optional[str] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    executed_tools: List[str] = field(default_factory=list)


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content**: For searching specific course content and detailed materials
2. **get_course_outline**: For getting course outlines, lesson lists, and course structure

Tool Usage Guidelines:
- **Up to 2 sequential tool calls allowed** - you can reason about results and make additional calls if needed
- **Course outline/structure queries**: Use get_course_outline tool to return course title, course link, and complete lesson list with numbers and titles
- **Content search queries**: Use search_course_content tool for specific material within courses
- **Sequential reasoning**: After receiving tool results, consider if additional tool calls would improve your answer
- **Strategic usage examples**:
  * First call: get_course_outline for a course → Then: search_course_content for specific concepts from that course
  * First call: search_course_content for general topic → Then: search_course_content with refined course/lesson filters
  * First call: search for one course → Then: search for comparison with another course
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool first, then format the response with course title, course link, and numbered lesson list
- **Course content questions**: Use search_course_content tool, potentially followed by refined searches
- **Multi-step reasoning**: If first tool results are insufficient, make a second targeted tool call
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Handle simple responses without tools
        if not tools or not tool_manager:
            return self._generate_simple_response(query, conversation_history)

        # Initialize round context for sequential processing
        context = RoundContext(
            original_query=query, conversation_history=conversation_history, tools=tools
        )

        # Start recursive processing with maximum 2 rounds
        return self._process_rounds_recursive(context, tool_manager, max_rounds=2)

    def _generate_simple_response(
        self, query: str, conversation_history: Optional[str] = None
    ) -> str:
        """Generate simple response without tools"""
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        response = self.client.messages.create(**api_params)
        return response.content[0].text

    def _process_rounds_recursive(
        self, context: RoundContext, tool_manager, max_rounds: int = 2
    ) -> str:
        """
        Recursively process rounds until termination condition is met.

        Termination conditions:
        1. Maximum rounds reached (2)
        2. Response has no tool_use blocks
        3. Tool execution fails
        """
        context.current_round += 1

        # Termination condition: max rounds reached
        if context.current_round > max_rounds:
            return self._handle_max_rounds_reached(context)

        # Get response for current round
        try:
            response = self._execute_round(context)
        except Exception as e:
            return self._handle_round_error(context, e)

        # Termination condition: no tool use
        if response.stop_reason != "tool_use":
            return self._extract_final_response(response)

        # Process tool calls and continue recursively
        tool_execution_result = self._execute_tools_for_round(
            response, context, tool_manager
        )

        # Termination condition: tool execution failed
        if tool_execution_result.failed:
            return self._handle_tool_execution_failure(tool_execution_result, context)

        # Update context with tool results and recurse
        self._update_context_with_tool_results(context, response, tool_execution_result)

        # Recursive call for next round
        return self._process_rounds_recursive(context, tool_manager, max_rounds)

    def _execute_round(self, context: RoundContext):
        """Execute a single round of AI interaction"""
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{context.conversation_history}"
            if context.conversation_history
            else self.SYSTEM_PROMPT
        )

        # Add round-specific guidance for subsequent rounds
        if context.current_round > 1:
            system_content += f"\n\nThis is round {context.current_round} of up to 2 rounds. Consider if additional tool calls would improve your answer based on previous results."

        api_params = {
            **self.base_params,
            "messages": context.messages.copy(),
            "system": system_content,
            "tools": context.tools,
            "tool_choice": {"type": "auto"},
        }

        return self.client.messages.create(**api_params)

    def _execute_tools_for_round(
        self, response, context: RoundContext, tool_manager
    ) -> ToolExecutionResult:
        """Execute all tool calls for the current round"""
        result = ToolExecutionResult()

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    result.tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                    result.executed_tools.append(content_block.name)

                except Exception as e:
                    result.failed = True
                    result.error_message = f"Tool execution failed: {str(e)}"
                    context.errors.append(
                        f"Round {context.current_round}: {result.error_message}"
                    )
                    break

        return result

    def _update_context_with_tool_results(
        self, context: RoundContext, response, tool_result: ToolExecutionResult
    ):
        """Update context with current round's results for next round"""
        # Add AI's tool use response
        context.messages.append({"role": "assistant", "content": response.content})

        # Add tool results
        if tool_result.tool_results:
            context.messages.append(
                {"role": "user", "content": tool_result.tool_results}
            )

    def _handle_max_rounds_reached(self, context: RoundContext) -> str:
        """Handle case where maximum rounds are reached"""
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{context.conversation_history}"
            if context.conversation_history
            else self.SYSTEM_PROMPT
        )
        system_content += (
            "\n\nProvide your final answer based on the tool results above."
        )

        final_params = {
            **self.base_params,
            "messages": context.messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _handle_tool_execution_failure(
        self, tool_result: ToolExecutionResult, context: RoundContext
    ) -> str:
        """Handle tool execution failures gracefully"""
        error_context = f"Tool execution failed in round {context.current_round}: {tool_result.error_message}"

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{context.conversation_history}"
            if context.conversation_history
            else self.SYSTEM_PROMPT
        )
        system_content += f"\n\nNote: {error_context}. Please provide the best answer you can based on available information."

        fallback_params = {
            **self.base_params,
            "messages": context.messages,
            "system": system_content,
        }

        try:
            fallback_response = self.client.messages.create(**fallback_params)
            return fallback_response.content[0].text
        except Exception as e:
            return f"I encountered an error while processing your request: {tool_result.error_message}"

    def _handle_round_error(self, context: RoundContext, error: Exception) -> str:
        """Handle errors during round execution"""
        error_msg = f"Error in round {context.current_round}: {str(error)}"
        context.errors.append(error_msg)
        return f"I encountered an error while processing your request. Please try rephrasing your question."

    def _extract_final_response(self, response) -> str:
        """Extract final response text from API response"""
        return response.content[0].text
