"""Graph builder abstraction + registry.

Each ``agent_type`` (chat / rag / workflow / ...) is implemented as a
:class:`GraphBuilder` subclass that assembles LangGraph nodes into a compiled
graph. Builders are registered globally via the :func:`register_graph`
decorator, and looked up by ``agent_type`` through :func:`get_graph_builder`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from baize.agent.models import AgentConfig

logger = logging.getLogger(__name__)


class GraphBuilder(ABC):
    """Abstract base for agent-type-specific graph builders.

    Subclasses register themselves via :func:`register_graph` and implement
    :meth:`build` to assemble a compiled LangGraph for their agent type.

    The ``thinking`` flag is forwarded into :meth:`build` per request — graphs
    decide which LLM nodes (if any) should have deep reasoning enabled via the
    inherited :meth:`_bind_thinking` helper.
    """

    #: ``agent_type`` value this builder handles. Set by ``@register_graph``.
    agent_type: str = ""

    @abstractmethod
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
        """Compile and return a LangGraph ready to execute.

        Args:
            llm: LangChain chat model (raw — call ``_bind_thinking`` before use).
            tools: LangChain tools resolved from agent_config.tools.
            agent_config: The full AgentConfig (for type-specific reads).
            thinking: Deep reasoning toggle for this request. True=enable,
                False=disable, None=use model default.
            checkpointer: Optional LangGraph checkpointer.
            services: Optional service map (e.g. ``{"retrieval": RetrievalService}``).
                Graph builders that require services (e.g. knowledge graph) read
                their dependencies from this dict; others ignore it.

        Returns:
            A compiled LangGraph instance (``CompiledGraph``).
        """

    @staticmethod
    def _bind_thinking(llm: Any, enabled: bool | None) -> Any:
        """Bind the thinking (deep reasoning) toggle to the LLM.

        Most providers (火山方舟 / Moonshot / Anthropic) only accept the binary
        form ``{"type": "enabled"}`` or ``{"type": "disabled"}``. OpenAI-style
        providers use ``reasoning_effort``.

        Args:
            llm: LangChain chat model instance (raw, before bind_tools).
            enabled: True = enable, False = disable, None = leave untouched.

        Returns:
            The LLM with bound kwargs, or the original LLM if ``enabled`` is None.
        """
        if enabled is None:
            return llm

        thinking = {"type": "enabled" if enabled else "disabled"}

        if llm.__class__.__name__ == "ChatAnthropic":
            try:
                return llm.bind(thinking=thinking)
            except Exception:
                logger.warning("Failed to bind thinking to Anthropic LLM; continuing.", exc_info=True)
                return llm

        if llm.__class__.__name__ == "ChatOpenAI":
            try:
                return llm.bind(reasoning_effort="medium" if enabled else "none")
            except Exception:
                logger.warning("Failed to bind reasoning to OpenAI LLM; continuing.", exc_info=True)
                return llm

        return llm


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_GRAPH_REGISTRY: dict[str, GraphBuilder] = {}


def register_graph(agent_type: str):
    """Decorator: register a :class:`GraphBuilder` subclass for an agent_type.

    Usage::

        @register_graph("chat")
        class ChatReactGraph(GraphBuilder):
            def build(self, ...): ...
    """

    def decorator(cls: type[GraphBuilder]) -> type[GraphBuilder]:
        instance = cls()
        instance.agent_type = agent_type
        _GRAPH_REGISTRY[agent_type] = instance
        return cls

    return decorator


def get_graph_builder(agent_type: str) -> GraphBuilder:
    """Look up a :class:`GraphBuilder` by ``agent_type``.

    Raises:
        ValueError: If ``agent_type`` is not registered. Hint at known types.
    """
    builder = _GRAPH_REGISTRY.get(agent_type)
    if builder is None:
        known = ", ".join(sorted(_GRAPH_REGISTRY.keys())) or "(none)"
        raise ValueError(f"Unknown agent_type: {agent_type!r}. Known: {known}")
    return builder


def list_agent_types() -> list[str]:
    """Return all registered agent types."""
    return sorted(_GRAPH_REGISTRY.keys())
