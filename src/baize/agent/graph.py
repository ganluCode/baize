"""ReAct Agent graph built with LangGraph.

Defines :class:`AgentState` and the :func:`build_react_graph` factory that
compiles a ReAct loop: LLM node → conditional edge → ToolNode → loop back.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import NotRequired
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the ReAct Agent graph.

    Attributes:
        messages: Conversation message history (accumulates via ``add_messages``).
        user_id: The current user's ID.
        agent_id: The agent configuration ID.
        shared_memory: Whether memories saved in this session are shared (visible
                       to all agents for the user) or private to the current agent.
                       Defaults to True when not provided.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    agent_id: str
    shared_memory: NotRequired[bool]


def _should_continue(state: AgentState) -> str:
    """Route after the LLM node: to tools if tool_calls present, else END."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_react_graph(
    llm: Any,
    tools: list[Any],
    checkpointer: Any | None = None,
):
    """Build and compile a ReAct Agent graph.

    Args:
        llm: A LangChain-compatible chat model (must support ``bind_tools``).
        tools: List of LangChain ``BaseTool`` instances to bind.
        checkpointer: Optional LangGraph checkpointer for state persistence.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for invocation.
    """
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Invoke the LLM with the current message history."""
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
