"""LangFuse observability integration for Baize.

Uses langfuse v2 native client with a TraceCollector pattern.
AgentService creates a collector at the start of each chat turn,
calls collector.add_*() during execution, and flush() at the end.
Business code never imports langfuse directly.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_langfuse_client = None


def get_langfuse_client(settings=None):
    """Return the singleton Langfuse client, or None if not configured."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if settings is None:
        from baize.core.config import get_settings
        settings = get_settings()

    if not settings.langfuse_public_key:
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("LangFuse client initialised: %s", settings.langfuse_host)
        return _langfuse_client
    except Exception:
        logger.warning("Failed to create LangFuse client; tracing disabled.", exc_info=True)
        return None


def create_langfuse_handler(settings=None):
    """Compatibility stub — returns None."""
    return None


# ---------------------------------------------------------------------------
# Span data classes — lightweight containers, no langfuse dependency
# ---------------------------------------------------------------------------


@dataclass
class LLMSpan:
    """Records one LLM call."""
    model: str = ""
    input: str = ""
    output: str = ""
    thinking: str = ""           # 推理内容（推理模型才有，非空时上报到 metadata）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


@dataclass
class ToolSpan:
    """Records one tool invocation."""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    latency_ms: int = 0


@dataclass
class RetrievalSpan:
    """Records one retrieval/search operation (RAG, memory recall, etc.)."""
    source: str = ""       # e.g. "memory", "rag", "web_search"
    query: str = ""
    results_count: int = 0
    latency_ms: int = 0


@dataclass
class SubAgentSpan:
    """Records one sub-agent delegation."""
    agent_id: str = ""
    agent_name: str = ""
    input: str = ""
    output: str = ""
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# TraceCollector — the only thing business code touches
# ---------------------------------------------------------------------------


class TraceCollector:
    """Real-time trace reporter for a single chat turn.

    Each span is sent to LangFuse **immediately** when added, so even if the
    process crashes mid-turn, already-reported spans are preserved.

    Usage in AgentService::

        collector = TraceCollector(user_id=..., session_id=..., agent_id=...)
        collector.set_input(message)

        # Spans are reported instantly
        collector.add_llm_span(model=..., output=..., ...)
        collector.add_tool_span(name=..., args=..., result=...)

        # Finalize trace with output (updates the root trace)
        collector.finalize(output=full_response)
        # Or on error:
        collector.finalize(output=partial_response, error="超时")
    """

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        agent_name: str = "",
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.message_id: str = ""
        self.input_message: str = ""
        self._start_time: float = time.monotonic()
        self._has_tool_spans: bool = False

        # Eagerly create the LangFuse trace (so child spans can attach to it)
        self._trace = None
        client = get_langfuse_client()
        if client is not None:
            try:
                self._trace = client.trace(
                    name="chat",
                    user_id=user_id,
                    session_id=session_id,
                    metadata={
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                    },
                    tags=[
                        f"agent:{agent_name}" if agent_name else f"agent:{agent_id[:8]}",
                    ],
                )
            except Exception:
                logger.debug("Failed to create LangFuse trace.", exc_info=True)

    def set_input(self, message: str) -> None:
        self.input_message = message
        if self._trace is not None:
            try:
                self._trace.update(input=message)
            except Exception:
                pass

    def add_llm_span(self, **kwargs) -> None:
        if self._trace is None:
            return
        span = LLMSpan(**kwargs)
        metadata: dict = {"latency_ms": span.latency_ms}
        if span.thinking:
            metadata["thinking"] = span.thinking
        try:
            self._trace.generation(
                name="llm",
                model=span.model,
                input=span.input,
                output=span.output,
                usage={"input": span.prompt_tokens, "output": span.completion_tokens},
                metadata=metadata,
            )
        except Exception:
            logger.debug("Failed to report LLM span.", exc_info=True)

    def add_tool_span(self, **kwargs) -> None:
        if self._trace is None:
            return
        self._has_tool_spans = True
        span = ToolSpan(**kwargs)
        try:
            self._trace.span(
                name=f"tool:{span.name}",
                input=span.args,
                output=span.result,
                metadata={"latency_ms": span.latency_ms},
            )
        except Exception:
            logger.debug("Failed to report tool span.", exc_info=True)

    def add_retrieval_span(self, **kwargs) -> None:
        if self._trace is None:
            return
        span = RetrievalSpan(**kwargs)
        try:
            self._trace.span(
                name=f"retrieval:{span.source}",
                input=span.query,
                output=f"{span.results_count} results",
                metadata={"latency_ms": span.latency_ms},
            )
        except Exception:
            logger.debug("Failed to report retrieval span.", exc_info=True)

    def add_sub_agent_span(self, **kwargs) -> None:
        if self._trace is None:
            return
        span = SubAgentSpan(**kwargs)
        try:
            self._trace.span(
                name=f"sub_agent:{span.agent_name or span.agent_id}",
                input=span.input,
                output=span.output,
                metadata={"latency_ms": span.latency_ms, "agent_id": span.agent_id},
            )
        except Exception:
            logger.debug("Failed to report sub-agent span.", exc_info=True)

    @property
    def total_latency_ms(self) -> int:
        return int((time.monotonic() - self._start_time) * 1000)

    def finalize(self, output: str = "", error: str | None = None) -> None:
        """Update the root trace with final output/error. Call once at the end."""
        if self._trace is None:
            return
        try:
            # Build final tags
            tags = [f"agent:{self.agent_name}" if self.agent_name else f"agent:{self.agent_id[:8]}"]
            if error:
                tags.append("error")
            if self._has_tool_spans:
                tags.append("tool_use")
            if self.total_latency_ms > 10000:
                tags.append("slow")

            self._trace.update(
                output=output,
                tags=tags,
                metadata={
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "message_id": self.message_id,
                    "total_latency_ms": self.total_latency_ms,
                },
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
        except Exception:
            logger.debug("Failed to finalize LangFuse trace.", exc_info=True)
