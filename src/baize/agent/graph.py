"""ReAct Agent graph built with LangGraph.

Defines :class:`AgentState` and the :func:`build_react_graph` factory that
compiles a ReAct loop: LLM node → conditional edge → ToolNode → loop back.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated, Any, NotRequired

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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


def _get_memory_service() -> Any | None:
    """Return the MemoryService from the global container, or None."""
    try:
        from baize.core.deps import get_container

        return get_container().memory_service
    except RuntimeError:
        return None


def build_react_graph(
    llm: Any,
    tools: list[Any],
    checkpointer: Any | None = None,
    auto_memory_recall: bool = False,
):
    """Build and compile a ReAct Agent graph.

    Args:
        llm: A LangChain-compatible chat model (must support ``bind_tools``).
        tools: List of LangChain ``BaseTool`` instances to bind.
        checkpointer: Optional LangGraph checkpointer for state persistence.
        auto_memory_recall: When True, search long-term memory before each LLM
            call and inject relevant memories into the System Prompt.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for invocation.
    """
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Invoke the LLM with the current message history."""
        messages: list[BaseMessage] = list(state["messages"])

        if auto_memory_recall:
            svc = _get_memory_service()
            if svc is not None:
                user_id: str = state.get("user_id", "")  # type: ignore[assignment]
                agent_id: str | None = state.get("agent_id")  # type: ignore[assignment]
                human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
                if human_msgs:
                    query = human_msgs[-1].content
                    try:
                        memories = await svc.search(
                            query, user_id=user_id, agent_id=agent_id, top_k=5
                        )
                    except Exception:  # noqa: BLE001
                        memories = []
                    if memories:
                        recall_lines = "\n".join(f"- {m.content}" for m in memories)
                        recall_section = f"\n\n## 记忆参考\n{recall_lines}"
                        if messages and isinstance(messages[0], SystemMessage):
                            messages[0] = SystemMessage(
                                content=messages[0].content + recall_section
                            )
                        else:
                            messages.insert(0, SystemMessage(content=recall_section.strip()))

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
