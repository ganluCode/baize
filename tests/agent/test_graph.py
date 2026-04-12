"""Tests for the ReAct Agent graph (F-010)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from baize.memory.interface import MemoryItem

_NOW = datetime(2026, 1, 1)


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


class TestAutoMemoryRecall:
    """Tests for auto_memory_recall integration in build_react_graph."""

    def _make_svc(self, memories: list[MemoryItem] | None = None) -> AsyncMock:
        svc = AsyncMock()
        svc.search.return_value = memories or []
        return svc

    async def test_auto_recall_injects_system_message_with_memories(self):
        """When auto_memory_recall=True and service returns memories, a SystemMessage is injected."""
        from baize.agent.graph import build_react_graph

        memories = [
            MemoryItem(id="m1", content="User loves Python", user_id="u-1", created_at=_NOW, updated_at=_NOW),
        ]
        svc = self._make_svc(memories)

        llm = _make_mock_llm([AIMessage(content="Got it.")])
        graph = build_react_graph(llm, [dummy_add], auto_memory_recall=True)

        captured_messages = []
        original_ainvoke = llm.ainvoke.side_effect

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            return next(iter(original_ainvoke))

        llm.ainvoke.side_effect = [AIMessage(content="Got it.")]

        with patch("baize.agent.graph._get_memory_service", return_value=svc):
            await graph.ainvoke({
                "messages": [HumanMessage(content="What do I like?")],
                "user_id": "u-1",
                "agent_id": "a-1",
            })

        svc.search.assert_called_once()
        call_kwargs = svc.search.call_args
        assert (
            call_kwargs.args[0] == "What do I like?"
            or call_kwargs.kwargs.get("query") == "What do I like?"
        )

    async def test_auto_recall_passes_user_id_and_agent_id_to_service(self):
        """search should be called with user_id and agent_id from state."""
        from baize.agent.graph import build_react_graph

        svc = self._make_svc([])
        llm = _make_mock_llm([AIMessage(content="ok")])
        graph = build_react_graph(llm, [], auto_memory_recall=True)

        with patch("baize.agent.graph._get_memory_service", return_value=svc):
            await graph.ainvoke({
                "messages": [HumanMessage(content="hello")],
                "user_id": "u-99",
                "agent_id": "a-42",
            })

        svc.search.assert_called_once()
        call_kwargs = svc.search.call_args
        assert call_kwargs.kwargs["user_id"] == "u-99"
        assert call_kwargs.kwargs["agent_id"] == "a-42"

    async def test_auto_recall_skips_when_false(self):
        """When auto_memory_recall=False, memory service should not be called."""
        from baize.agent.graph import build_react_graph

        svc = self._make_svc([])
        llm = _make_mock_llm([AIMessage(content="ok")])
        graph = build_react_graph(llm, [], auto_memory_recall=False)

        with patch("baize.agent.graph._get_memory_service", return_value=svc) as mock_get:
            await graph.ainvoke({
                "messages": [HumanMessage(content="hello")],
                "user_id": "u-1",
                "agent_id": "a-1",
            })

        mock_get.assert_not_called()

    async def test_auto_recall_no_system_message_when_no_memories(self):
        """When service returns empty, LLM should be called without injected SystemMessage."""
        from baize.agent.graph import build_react_graph

        svc = self._make_svc([])
        final_msg = AIMessage(content="ok")
        llm = _make_mock_llm([final_msg])
        graph = build_react_graph(llm, [], auto_memory_recall=True)

        with patch("baize.agent.graph._get_memory_service", return_value=svc):
            result = await graph.ainvoke({
                "messages": [HumanMessage(content="hello")],
                "user_id": "u-1",
                "agent_id": "a-1",
            })

        # No SystemMessage should appear in the final messages
        system_msgs = [m for m in result["messages"] if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 0

    async def test_auto_recall_appends_to_existing_system_message(self):
        """When a SystemMessage already exists, recall is appended to it."""
        from baize.agent.graph import build_react_graph

        memories = [
            MemoryItem(id="m1", content="User is an engineer", user_id="u-1", created_at=_NOW, updated_at=_NOW),
        ]
        svc = self._make_svc(memories)

        # Track what messages the LLM actually receives
        received: list[list] = []

        async def capturing_ainvoke(msgs):
            received.append(list(msgs))
            return AIMessage(content="ok")

        llm = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.ainvoke = capturing_ainvoke

        graph = build_react_graph(llm, [], auto_memory_recall=True)

        with patch("baize.agent.graph._get_memory_service", return_value=svc):
            await graph.ainvoke({
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="Who am I?"),
                ],
                "user_id": "u-1",
                "agent_id": "a-1",
            })

        assert len(received) == 1
        first_msg = received[0][0]
        assert isinstance(first_msg, SystemMessage)
        assert "You are a helpful assistant." in first_msg.content
        assert "User is an engineer" in first_msg.content

    async def test_auto_recall_skips_when_service_unavailable(self):
        """When memory service is None, graph should still work normally."""
        from baize.agent.graph import build_react_graph

        llm = _make_mock_llm([AIMessage(content="ok")])
        graph = build_react_graph(llm, [], auto_memory_recall=True)

        with patch("baize.agent.graph._get_memory_service", return_value=None):
            result = await graph.ainvoke({
                "messages": [HumanMessage(content="hello")],
                "user_id": "u-1",
                "agent_id": "a-1",
            })

        assert result["messages"][-1].content == "ok"
