"""Abstract pipeline step with automatic tracing — Template Method pattern.

A "step" is a stage in Baize's agent execution pipeline (context preparation,
graph execution, persistence). Not to be confused with LangGraph **nodes**,
which are the building blocks of a single graph — those live in
:mod:`baize.agent.graphs.nodes`.

Every step inherits from :class:`BaseStep`:
  - ``_execute()``  → subclass implements core logic
  - ``_report()``   → subclass defines how to report its span
  - ``run()``       → base class orchestrates: time → execute → report

Subclasses never import langfuse or call :class:`TraceCollector` directly.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from baize.core.observability import TraceCollector

logger = logging.getLogger(__name__)


@dataclass
class StepContext:
    """Shared context passed through the pipeline.

    Carries references needed by all steps (collector, user, agent, etc.).
    Steps can read from it and write results to ``results``.
    """

    collector: "TraceCollector"
    user_id: str = ""
    session_id: str = ""
    agent_id: str = ""
    message: str = ""
    # Steps can store results here for downstream steps to pick up
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Standard return from a step execution."""

    success: bool = True
    data: Any = None
    error: str | None = None


class BaseStep(ABC):
    """Abstract pipeline step with automatic timing and trace reporting.

    Subclasses must implement:
        ``_execute(ctx)`` — the actual work
        ``_report(ctx, result, latency_ms)`` — how to report the span

    Optionally override:
        ``step_name`` — display name for logs and traces (default: class name)
    """

    @property
    def step_name(self) -> str:
        return self.__class__.__name__

    async def run(self, ctx: StepContext) -> StepResult:
        """Template method: time → execute → report. Never override this."""
        start = time.monotonic()
        try:
            result = await self._execute(ctx)
            latency_ms = int((time.monotonic() - start) * 1000)
            self._report(ctx, result, latency_ms)
            logger.debug("Step %s completed in %dms", self.step_name, latency_ms)
            return result
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            error_result = StepResult(success=False, error=str(exc))
            self._report(ctx, error_result, latency_ms)
            logger.warning("Step %s failed after %dms: %s", self.step_name, latency_ms, exc)
            raise

    @abstractmethod
    async def _execute(self, ctx: StepContext) -> StepResult:
        """Subclass implements core logic here. Only business code, no tracing."""
        ...

    @abstractmethod
    def _report(self, ctx: StepContext, result: StepResult, latency_ms: int) -> None:
        """Subclass defines how to report its span to the collector."""
        ...
