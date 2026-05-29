"""LangGraph builders organized by agent_type.

Each graph file (chat_react.py, rag.py, ...) registers a :class:`GraphBuilder`
via the ``@register_graph(agent_type)`` decorator. Import this package to
trigger registration of all builders.

Usage::

    from baize.agent.graphs import get_graph_builder
    builder = get_graph_builder(agent_config.agent_type)
    graph = builder.build(llm=llm, tools=tools, agent_config=agent_config)
"""

# Import each graph module to trigger @register_graph side-effects.
from baize.agent.graphs import chat_react  # noqa: F401
from baize.agent.graphs import knowledge_react  # noqa: F401
from baize.agent.graphs.base import GraphBuilder, get_graph_builder, list_agent_types, register_graph
from baize.agent.graphs.state import AgentState

__all__ = [
    "AgentState",
    "GraphBuilder",
    "get_graph_builder",
    "list_agent_types",
    "register_graph",
]
