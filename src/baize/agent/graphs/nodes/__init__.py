"""LangGraph nodes — reusable building blocks for agent graphs.

Each node is a factory function returning an async callable conforming to the
LangGraph node signature ``(state) -> dict``. Graphs in ``graphs/`` assemble
these nodes via ``StateGraph.add_node(name, factory_result)``.
"""

from baize.agent.graphs.nodes.llm_node import build_llm_node, should_continue_with_tools

__all__ = ["build_llm_node", "should_continue_with_tools"]
