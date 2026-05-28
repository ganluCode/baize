"""Chat ReAct graph — the default conversational agent.

Pattern: LLM ↔ ToolNode loop.

  START → llm_node ──(no tool_calls)──→ END
              │
              └──(tool_calls)──→ tools_node ──→ llm_node (loop)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from baize.agent.graphs.base import GraphBuilder, register_graph
from baize.agent.graphs.nodes import build_llm_node, should_continue_with_tools
from baize.agent.graphs.state import AgentState

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from baize.agent.models import AgentConfig


@register_graph("chat")
class ChatReactGraph(GraphBuilder):
    """Standard ReAct agent: LLM with tool-calling loop."""

    def build(
        self,
        *,
        llm: Any,
        tools: list["BaseTool"],
        agent_config: "AgentConfig",
        thinking: bool | None = None,
        checkpointer: Any | None = None,
    ) -> Any:
        llm = self._bind_thinking(llm, thinking)
        llm_node = build_llm_node(llm, tools, auto_memory_recall=False)
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
