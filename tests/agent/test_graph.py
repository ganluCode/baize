"""Tests for the ReAct Agent graph (F-010)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@lc_tool
def dummy_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@lc_tool
def failing_tool(x: str) -> str:
    """A tool that always raises."""
    raise RuntimeError("boom")


def _make_mock_llm(responses: list[AIMessage]):
    """Create a mock LLM that returns *responses* in order.

    The mock supports `bind_tools` (returns self) and `ainvoke`.
    """
    llm = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(side_effect=responses)
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildReactGraph:
    """Tests for build_react_graph factory function."""

    def test_returns_compiled_graph(self):
        from baize.agent.graph import build_react_graph

        llm = _make_mock_llm([])
        graph = build_react_graph(llm, [dummy_add])
        # Should return a CompiledGraph (has invoke / astream_events)
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream_events")

    def test_binds_tools_to_llm(self):
        from baize.agent.graph import build_react_graph

        llm = _make_mock_llm([])
        build_react_graph(llm, [dummy_add, failing_tool])
        llm.bind_tools.assert_called_once_with([dummy_add, failing_tool])

    async def test_direct_response_no_tool_calls(self):
        """LLM responds without tool_calls → graph should end immediately."""
        from baize.agent.graph import build_react_graph

        final_msg = AIMessage(content="Hello!")
        llm = _make_mock_llm([final_msg])
        graph = build_react_graph(llm, [dummy_add])

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Hi")]},
        )
        # The last message should be the AIMessage
        assert result["messages"][-1].content == "Hello!"

    async def test_tool_call_then_response(self):
        """LLM makes a tool_call, tool executes, LLM responds with final answer."""
        from baize.agent.graph import build_react_graph

        # First response: tool call
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "dummy_add", "args": {"a": 2, "b": 3}}],
        )
        # Second response: final answer (after tool result)
        final_msg = AIMessage(content="The answer is 5.")

        llm = _make_mock_llm([tool_call_msg, final_msg])
        graph = build_react_graph(llm, [dummy_add])

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is 2+3?")]},
        )

        messages = result["messages"]
        # Should have: HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)
        assert len(messages) == 4
        assert isinstance(messages[0], HumanMessage)
        assert messages[1].tool_calls  # AIMessage with tool_calls
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].content == "5"  # dummy_add returns int, converted to str
        assert messages[-1].content == "The answer is 5."

    async def test_tool_exception_captured_as_error_message(self):
        """When a tool raises, ToolNode should capture the error in a ToolMessage."""
        from baize.agent.graph import build_react_graph

        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_err", "name": "failing_tool", "args": {"x": "test"}}],
        )
        final_msg = AIMessage(content="Sorry, the tool failed.")

        llm = _make_mock_llm([tool_call_msg, final_msg])
        graph = build_react_graph(llm, [failing_tool])

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Do something")]},
        )

        messages = result["messages"]
        # Find the ToolMessage — it should contain the error
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "Error" in tool_msgs[0].content or "boom" in tool_msgs[0].content

    async def test_state_contains_user_id_and_agent_id(self):
        """AgentState should pass through user_id and agent_id."""
        from baize.agent.graph import build_react_graph

        final_msg = AIMessage(content="ok")
        llm = _make_mock_llm([final_msg])
        graph = build_react_graph(llm, [dummy_add])

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="test")],
                "user_id": "u-123",
                "agent_id": "a-456",
            },
        )
        assert result["user_id"] == "u-123"
        assert result["agent_id"] == "a-456"

    async def test_astream_events_produces_events(self):
        """Graph should support astream_events for streaming."""
        from baize.agent.graph import build_react_graph

        final_msg = AIMessage(content="Streamed!")
        llm = _make_mock_llm([final_msg])
        graph = build_react_graph(llm, [dummy_add])

        events = []
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content="Stream test")]},
            version="v2",
        ):
            events.append(event)

        # Should produce at least some events
        assert len(events) > 0

    def test_checkpointer_passed_to_compile(self):
        """When checkpointer is provided, it should be passed to compile."""
        from langgraph.checkpoint.memory import InMemorySaver

        from baize.agent.graph import build_react_graph

        llm = _make_mock_llm([])
        checkpointer = InMemorySaver()
        graph = build_react_graph(llm, [dummy_add], checkpointer=checkpointer)
        # Graph should still be valid
        assert hasattr(graph, "ainvoke")
