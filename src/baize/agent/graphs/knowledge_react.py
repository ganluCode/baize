"""Knowledge graph — RAG-style agent: retrieve → answer → END.

Pattern: single-pass retrieval then LLM answer with citations.

  START → retrieve_node → answer_node → END
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph

from baize.agent.graphs.base import GraphBuilder, register_graph
from baize.agent.graphs.nodes.knowledge_answer import build_knowledge_answer_node
from baize.agent.graphs.nodes.knowledge_retrieve import build_knowledge_retrieve_node
from baize.agent.graphs.state import AgentState

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from baize.agent.models import AgentConfig


@register_graph("knowledge")
class KnowledgeGraph(GraphBuilder):
    """Knowledge retrieval agent: retrieve relevant chunks then generate a cited answer.

    Tools are intentionally ignored in the first period — the graph uses a fixed
    retrieve → answer pipeline with no tool-calling loop.
    """

    def build(
        self,
        *,
        llm: Any,
        tools: list["BaseTool"],
        agent_config: "AgentConfig",
        thinking: bool | None = None,
        checkpointer: Any | None = None,
        services: dict[str, Any] | None = None,
    ) -> Any:
        """Compile a knowledge-retrieval LangGraph.

        Args:
            llm: LangChain chat model for the answer node.
            tools: Ignored — knowledge graph does not use tools in first period.
            agent_config: Agent configuration; must have a valid ``knowledge_config``.
            thinking: Deep reasoning toggle forwarded to the LLM.
            checkpointer: Optional LangGraph checkpointer.
            services: Must contain ``"retrieval"`` key with a ``RetrievalService``.

        Returns:
            A compiled LangGraph with nodes: retrieve → answer → END.

        Raises:
            ValueError: If ``services["retrieval"]`` is missing.
            ValueError: If ``agent_config.knowledge_config`` is None or has no
                ``default_kb_id``.
        """
        if not services or "retrieval" not in services:
            raise ValueError(
                "KnowledgeGraph requires services['retrieval'] (a RetrievalService instance); "
                f"got services={services!r}"
            )
        retrieval_svc = services["retrieval"]

        knowledge_config: dict | None = agent_config.knowledge_config
        if knowledge_config is None:
            raise ValueError(
                "KnowledgeGraph requires agent_config.knowledge_config to be set; got None"
            )

        default_kb_id = knowledge_config.get("default_kb_id")
        if not default_kb_id:
            raise ValueError(
                "KnowledgeGraph requires agent_config.knowledge_config['default_kb_id']; "
                f"got knowledge_config={knowledge_config!r}"
            )

        kb_id = uuid.UUID(str(default_kb_id))
        top_k = int(knowledge_config.get("top_k", 8))
        include_parents = bool(knowledge_config.get("include_parents", True))

        llm = self._bind_thinking(llm, thinking)

        retrieve_node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=kb_id,
            top_k=top_k,
            include_parents=include_parents,
        )
        answer_node = build_knowledge_answer_node(llm=llm)

        graph = StateGraph(AgentState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("answer", answer_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)

        return graph.compile(checkpointer=checkpointer)
