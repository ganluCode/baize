"""LLM execution node — reusable across all agent types.

Calls the bound LLM with the current message history, optionally injecting
recalled memories into the system prompt when ``auto_memory_recall=True``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from baize.agent.graphs.state import AgentState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _get_memory_service() -> Any | None:
    """Return the MemoryService from the global container, or None."""
    try:
        from baize.core.deps import get_container

        return get_container().memory_service
    except RuntimeError:
        return None


def build_llm_node(
    llm: "BaseLanguageModel",
    tools: list["BaseTool"],
    *,
    auto_memory_recall: bool = False,
) -> Callable[["AgentState"], Awaitable[dict[str, list[BaseMessage]]]]:
    """Build an async LLM node callable for use with LangGraph.

    Args:
        llm: LangChain chat model (must support ``bind_tools``).
        tools: Tools to bind to the LLM.
        auto_memory_recall: When True, search long-term memory before each LLM
            call and inject relevant memories into the System Prompt.

    Returns:
        An async function ``(state) -> {"messages": [response]}`` suitable for
        :meth:`StateGraph.add_node`.
    """
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    async def llm_node(state: "AgentState") -> dict[str, list[BaseMessage]]:
        messages: list[BaseMessage] = list(state["messages"])

        if auto_memory_recall:
            svc = _get_memory_service()
            if svc is not None:
                user_id: str = state.get("user_id", "")
                agent_id: str | None = state.get("agent_id")
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

    return llm_node


def should_continue_with_tools(state: "AgentState") -> str:
    """Conditional edge: route to ``tools`` if last AIMessage has tool_calls, else END."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END
