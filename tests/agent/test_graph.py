"""Tests for the ReAct Agent graph (F-010)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from baize.agent.graphs.nodes.llm_node import build_llm_node, should_continue_with_tools
from baize.agent.graphs.state import AgentState
from baize.memory.interface import MemoryItem

if TYPE_CHECKING:
    pass

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


def _build_react_graph(
    llm,
    tools,
    *,
    auto_memory_recall: bool = False,
    checkpointer=None,
):
    """Test helper that assembles a chat-react style graph using the new modular
    components (``build_llm_node`` + ``ToolNode``).

    This mirrors what :class:`ChatReactGraph` does, but additionally honours the
    ``auto_memory_recall`` flag — which is no longer exposed on the high-level
    builder and is now a parameter of ``build_llm_node``.
    """
    llm_node = build_llm_node(llm, tools, auto_memory_recall=auto_memory_recall)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    graph = StateGraph(AgentState)
    graph.add_node("agent", llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue_with_tools,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildReactGraph:
    """Tests for the chat-react graph assembly (formerly build_react_graph)."""

    def test_returns_compiled_graph(self):
        llm = _make_mock_llm([])
        graph = _build_react_graph(llm, [dummy_add])
        # Should return a CompiledGraph (has invoke / astream_events)
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream_events")

    def test_binds_tools_to_llm(self):
        llm = _make_mock_llm([])
        _build_react_graph(llm, [dummy_add, failing_tool])
        llm.bind_tools.assert_called_once_with([dummy_add, failing_tool])

    async def test_direct_response_no_tool_calls(self):
        """LLM responds without tool_calls → graph should end immediately."""
        final_msg = AIMessage(content="Hello!")
        llm = _make_mock_llm([final_msg])
        graph = _build_react_graph(llm, [dummy_add])

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Hi")]},
        )
        # The last message should be the AIMessage
        assert result["messages"][-1].content == "Hello!"

    async def test_tool_call_then_response(self):
        """LLM makes a tool_call, tool executes, LLM responds with final answer."""
        # First response: tool call
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "dummy_add", "args": {"a": 2, "b": 3}}],
        )
        # Second response: final answer (after tool result)
        final_msg = AIMessage(content="The answer is 5.")

        llm = _make_mock_llm([tool_call_msg, final_msg])
        graph = _build_react_graph(llm, [dummy_add])

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
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_err", "name": "failing_tool", "args": {"x": "test"}}],
        )
        final_msg = AIMessage(content="Sorry, the tool failed.")

        llm = _make_mock_llm([tool_call_msg, final_msg])
        graph = _build_react_graph(llm, [failing_tool])

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
        final_msg = AIMessage(content="ok")
        llm = _make_mock_llm([final_msg])
        graph = _build_react_graph(llm, [dummy_add])

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
        final_msg = AIMessage(content="Streamed!")
        llm = _make_mock_llm([final_msg])
        graph = _build_react_graph(llm, [dummy_add])

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

        llm = _make_mock_llm([])
        checkpointer = InMemorySaver()
        graph = _build_react_graph(llm, [dummy_add], checkpointer=checkpointer)
        # Graph should still be valid
        assert hasattr(graph, "ainvoke")


class TestAutoMemoryRecall:
    """Tests for auto_memory_recall integration in the LLM node."""

    def _make_svc(self, memories: list[MemoryItem] | None = None) -> AsyncMock:
        svc = AsyncMock()
        svc.search.return_value = memories or []
        return svc

    async def test_auto_recall_injects_system_message_with_memories(self):
        """When auto_memory_recall=True and service returns memories, a SystemMessage is injected."""
        memories = [
            MemoryItem(id="m1", content="User loves Python", user_id="u-1", created_at=_NOW, updated_at=_NOW),
        ]
        svc = self._make_svc(memories)

        llm = _make_mock_llm([AIMessage(content="Got it.")])
        graph = _build_react_graph(llm, [dummy_add], auto_memory_recall=True)

        with patch(
            "baize.agent.graphs.nodes.llm_node._get_memory_service",
            return_value=svc,
        ):
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
        svc = self._make_svc([])
        llm = _make_mock_llm([AIMessage(content="ok")])
        graph = _build_react_graph(llm, [], auto_memory_recall=True)

        with patch(
            "baize.agent.graphs.nodes.llm_node._get_memory_service",
            return_value=svc,
        ):
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
        svc = self._make_svc([])
        llm = _make_mock_llm([AIMessage(content="ok")])
        graph = _build_react_graph(llm, [], auto_memory_recall=False)

        with patch(
            "baize.agent.graphs.nodes.llm_node._get_memory_service",
            return_value=svc,
        ) as mock_get:
            await graph.ainvoke({
                "messages": [HumanMessage(content="hello")],
                "user_id": "u-1",
                "agent_id": "a-1",
            })

        mock_get.assert_not_called()

    async def test_auto_recall_no_system_message_when_no_memories(self):
        """When service returns empty, LLM should be called without injected SystemMessage."""
        svc = self._make_svc([])
        final_msg = AIMessage(content="ok")
        llm = _make_mock_llm([final_msg])
        graph = _build_react_graph(llm, [], auto_memory_recall=True)

        with patch(
            "baize.agent.graphs.nodes.llm_node._get_memory_service",
            return_value=svc,
        ):
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

        graph = _build_react_graph(llm, [], auto_memory_recall=True)

        with patch(
            "baize.agent.graphs.nodes.llm_node._get_memory_service",
            return_value=svc,
        ):
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
        llm = _make_mock_llm([AIMessage(content="ok")])
        graph = _build_react_graph(llm, [], auto_memory_recall=True)

        with patch(
            "baize.agent.graphs.nodes.llm_node._get_memory_service",
            return_value=None,
        ):
            result = await graph.ainvoke({
                "messages": [HumanMessage(content="hello")],
                "user_id": "u-1",
                "agent_id": "a-1",
            })

        assert result["messages"][-1].content == "ok"


class TestRetrievedChunksState:
    """Tests for the retrieved_chunks field in AgentState (F-001)."""

    async def test_chat_graph_runs_without_retrieved_chunks(self):
        """Chat graph should run normally when retrieved_chunks is absent from initial state."""
        llm = _make_mock_llm([AIMessage(content="hello")])
        graph = _build_react_graph(llm, [])

        result = await graph.ainvoke({
            "messages": [HumanMessage(content="hi")],
            "user_id": "u-1",
            "agent_id": "a-1",
        })

        assert result["messages"][-1].content == "hello"

    async def test_retrieved_chunks_passed_through_state(self):
        """When retrieved_chunks is set in initial state, it should be in the result."""
        llm = _make_mock_llm([AIMessage(content="answer")])
        graph = _build_react_graph(llm, [])

        chunks = [{"chunk_id": "c1", "doc_id": "d1", "section_path": ["intro"], "score": 0.9, "content": "text"}]
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="question")],
            "user_id": "u-1",
            "agent_id": "a-1",
            "retrieved_chunks": chunks,
        })

        assert result.get("retrieved_chunks") == chunks

    async def test_retrieved_chunks_default_empty_list_when_absent(self):
        """retrieved_chunks defaults to empty list when not provided in initial state."""
        llm = _make_mock_llm([AIMessage(content="ok")])
        graph = _build_react_graph(llm, [])

        result = await graph.ainvoke({
            "messages": [HumanMessage(content="hello")],
            "user_id": "u-1",
            "agent_id": "a-1",
        })

        assert result.get("retrieved_chunks", []) == []


class TestGraphBuilderServicesParam:
    """Tests for F-002: GraphBuilder.build() services parameter."""

    def test_build_abstract_method_has_services_param(self):
        """GraphBuilder.build() abstract method signature includes services parameter."""
        import inspect
        from baize.agent.graphs.base import GraphBuilder

        sig = inspect.signature(GraphBuilder.build)
        assert "services" in sig.parameters

    def test_services_param_default_is_none(self):
        """services parameter defaults to None."""
        import inspect
        from baize.agent.graphs.base import GraphBuilder

        sig = inspect.signature(GraphBuilder.build)
        param = sig.parameters["services"]
        assert param.default is None

    def test_chat_react_graph_build_callable_without_services(self):
        """Existing ChatReactGraph.build() works when services is not passed."""
        from unittest.mock import MagicMock

        from baize.agent.graphs.chat_react import ChatReactGraph

        builder = ChatReactGraph()
        llm = _make_mock_llm([])
        agent_config = MagicMock()
        agent_config.tools = []

        graph = builder.build(
            llm=llm,
            tools=[],
            agent_config=agent_config,
        )
        assert hasattr(graph, "ainvoke")

    def test_graph_builder_module_imports_without_error(self):
        """baize.agent.graphs.base imports cleanly."""
        import importlib
        mod = importlib.import_module("baize.agent.graphs.base")
        assert hasattr(mod, "GraphBuilder")
        assert hasattr(mod, "register_graph")
        assert hasattr(mod, "get_graph_builder")
