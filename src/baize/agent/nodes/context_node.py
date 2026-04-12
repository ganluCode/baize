"""Context preparation node — memory recall + prompt assembly + history compression."""

from __future__ import annotations

from typing import TYPE_CHECKING

from baize.agent.nodes.base import BaseNode, NodeContext, NodeResult

if TYPE_CHECKING:
    from baize.context.service import ContextService, PreparedContext
    from baize.agent.models import AgentConfig
    from baize.user.models import UserModel


class ContextNode(BaseNode):
    """Prepares the full context for a chat turn.

    Reads agent config, user profile, session history, and memory.
    Stores the PreparedContext in ``ctx.results["prepared"]``.
    """

    node_name = "context"

    def __init__(
        self,
        context_svc: "ContextService",
        agent_config: "AgentConfig",
        user: "UserModel",
        llm,
    ) -> None:
        self._context_svc = context_svc
        self._agent_config = agent_config
        self._user = user
        self._llm = llm

    async def _execute(self, ctx: NodeContext) -> NodeResult:
        import uuid

        prepared = await self._context_svc.prepare(
            agent_config=self._agent_config,
            user=self._user,
            session_id=uuid.UUID(ctx.session_id),
            message=ctx.message,
            llm=self._llm,
        )
        ctx.results["prepared"] = prepared
        return NodeResult(data=prepared)

    def _report(self, ctx: NodeContext, result: NodeResult, latency_ms: int) -> None:
        prepared = result.data if result.success else None
        memory_count = len(prepared.memories) if prepared and hasattr(prepared, "memories") else 0
        ctx.collector.add_retrieval_span(
            source="context",
            query=ctx.message,
            results_count=memory_count,
            latency_ms=latency_ms,
        )
