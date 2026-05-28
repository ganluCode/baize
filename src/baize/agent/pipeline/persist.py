"""Message persistence step — saves user and assistant messages to DB."""

from __future__ import annotations

from typing import TYPE_CHECKING

from baize.agent.pipeline.base import BaseStep, StepContext, StepResult
from baize.session.models import MessageRole

if TYPE_CHECKING:
    import uuid

    from baize.session.service import SessionService


class PersistStep(BaseStep):
    """Persists user + assistant messages and stores message_id in context."""

    step_name = "persist"

    def __init__(
        self,
        session_svc: "SessionService",
        session_id: "uuid.UUID",
        user_id: "uuid.UUID",
    ) -> None:
        self._session_svc = session_svc
        self._session_id = session_id
        self._user_id = user_id

    async def _execute(self, ctx: StepContext) -> StepResult:
        response = ctx.results.get("response", "")
        thinking = ctx.results.get("thinking")

        user_msg = await self._session_svc.save_message(
            session_id=self._session_id,
            user_id=self._user_id,
            role=MessageRole.user,
            content=ctx.message,
        )
        await self._session_svc.save_message(
            session_id=self._session_id,
            user_id=self._user_id,
            role=MessageRole.assistant,
            content=response,
            thinking=thinking,
        )

        ctx.results["message_id"] = str(user_msg.id)
        return StepResult(data={"message_id": str(user_msg.id)})

    def _report(self, ctx: StepContext, result: StepResult, latency_ms: int) -> None:
        # Persistence is an internal operation, no need for a visible span.
        # But we update the collector with the message_id.
        if result.success and result.data:
            ctx.collector.message_id = result.data.get("message_id", "")
