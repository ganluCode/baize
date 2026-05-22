"""Abstract base node with automatic tracing — Template Method pattern.

Every step in the agent pipeline inherits from BaseNode:
  - _execute()   → subclass implements core logic
  - _report()    → subclass defines how to report its span
  - run()        → base class orchestrates: time → execute → report

Subclasses never import langfuse or call TraceCollector directly.
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
class NodeContext:
    """Shared context passed through the node pipeline.

    Carries references needed by all nodes (collector, user, agent, etc.).
    Nodes can read from it and write results to ``results``.
    """

    collector: "TraceCollector"
    user_id: str = ""
    session_id: str = ""
    agent_id: str = ""
    message: str = ""
    # Nodes can store results here for downstream nodes to pick up
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    """Standard return from a node execution."""

    success: bool = True
    data: Any = None
    error: str | None = None


class BaseNode(ABC):
    """Abstract pipeline node with automatic timing and trace reporting.

    Subclasses must implement:
        ``_execute(ctx)`` — the actual work
        ``_report(ctx, result, latency_ms)`` — how to report the span

    Optionally override:
        ``node_name`` — display name for logs and traces (default: class name)
    """

    @property
    def node_name(self) -> str:
        return self.__class__.__name__

    async def run(self, ctx: NodeContext) -> NodeResult:
        """Template method: time → execute → report. Never override this."""
        start = time.monotonic()
        try:
            result = await self._execute(ctx)
            latency_ms = int((time.monotonic() - start) * 1000)
            self._report(ctx, result, latency_ms)
            logger.debug("Node %s completed in %dms", self.node_name, latency_ms)
            return result
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            error_result = NodeResult(success=False, error=str(exc))
            self._report(ctx, error_result, latency_ms)
            logger.warning("Node %s failed after %dms: %s", self.node_name, latency_ms, exc)
            raise

    @abstractmethod
    async def _execute(self, ctx: NodeContext) -> NodeResult:
        """Subclass implements core logic here. Only business code, no tracing."""
        ...

    @abstractmethod
    def _report(self, ctx: NodeContext, result: NodeResult, latency_ms: int) -> None:
        """Subclass defines how to report its span to the collector."""
        ...
