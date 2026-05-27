"""ReAct execution node — wraps the LangGraph ReAct loop with automatic tracing.

Streams LangGraph events and yields ChatEvents, while reporting LLM and tool
spans to the TraceCollector in real time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from baize.agent.graph import build_react_graph
from baize.agent.nodes.base import BaseNode, NodeContext, NodeResult
from baize.agent.service import ChatEvent
from baize.agent.tools import ToolRegistry

if TYPE_CHECKING:
    from baize.agent.models import AgentConfig
    from baize.context.service import PreparedContext

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 10
_CHAT_TIMEOUT = 120.0


def _apply_thinking_to_llm(llm: Any, enabled: bool | None) -> Any:
    """Bind the thinking (deep reasoning) toggle to the LLM.

    Thinking is a binary on/off switch — most providers (火山方舟 / Moonshot)
    only accept ``{"type": "enabled"}`` or ``{"type": "disabled"}``.

    Args:
        llm: LangChain chat model instance.
        enabled: True = enable thinking, False = disable, None = use model default.

    Returns:
        LLM with bound thinking kwarg, or the original LLM when enabled is None.
    """
    if enabled is None:
        # Don't touch — let the model use its own default
        return llm

    thinking = {"type": "enabled" if enabled else "disabled"}

    # 火山方舟 / Moonshot / Anthropic 都走 Anthropic 协议的 thinking 字段
    if llm.__class__.__name__ == "ChatAnthropic":
        try:
            return llm.bind(thinking=thinking)
        except Exception:
            logger.warning("Failed to bind thinking to Anthropic LLM; continuing.", exc_info=True)
            return llm

    # OpenAI 协议：用 reasoning_effort，none = 关闭
    if llm.__class__.__name__ == "ChatOpenAI":
        try:
            return llm.bind(reasoning_effort="medium" if enabled else "none")
        except Exception:
            logger.warning("Failed to bind reasoning to OpenAI LLM; continuing.", exc_info=True)
            return llm

    return llm


class ReactNode(BaseNode):
    """Executes the LangGraph ReAct loop, streaming ChatEvents.

    Unlike other nodes, this one is a generator — call ``stream()``
    instead of ``run()`` to get ChatEvents. Tracing is still automatic.
    """

    node_name = "react"

    def __init__(
        self,
        llm,
        agent_config: "AgentConfig",
        thinking: bool | None = None,
    ) -> None:
        self._llm = llm
        self._agent_config = agent_config
        self._model_name = getattr(llm, "model_name", "") or getattr(llm, "model", "")
        self._thinking = thinking

    async def stream(self, ctx: NodeContext) -> AsyncIterator[ChatEvent]:
        """Stream ChatEvents from the ReAct loop with automatic tracing.

        This replaces ``run()`` for this node because it's a generator.
        """
        start = time.monotonic()
        prepared: "PreparedContext" = ctx.results["prepared"]

        # Resolve tools
        tool_names: list[str] = []
        if self._agent_config.tools and isinstance(self._agent_config.tools, dict):
            tool_names.extend(self._agent_config.tools.get("builtin", []))
        elif self._agent_config.tools and isinstance(self._agent_config.tools, list):
            tool_names.extend(self._agent_config.tools)
        tool_entries = ToolRegistry.get_by_names(tool_names)
        tools = [e.langchain_tool for e in tool_entries if e.langchain_tool is not None]

        # Build graph — apply reasoning/thinking kwargs if requested
        llm = _apply_thinking_to_llm(self._llm, self._thinking)
        graph = build_react_graph(llm, tools)

        # Initial messages
        initial_messages = [
            SystemMessage(content=prepared.system_prompt),
            *prepared.history,
            HumanMessage(content=ctx.message),
        ]

        run_config: dict = {
            "configurable": {"thread_id": ctx.session_id},
            "metadata": {
                "user_id": ctx.user_id,
                "session_id": ctx.session_id,
                "agent_id": ctx.agent_id,
            },
        }

        tool_call_count = 0
        current_response = ""
        current_thinking = ""
        _llm_start_time: float = 0.0
        _tool_start_times: dict[str, float] = {}
        _pending_tool_args: dict[str, dict] = {}

        try:
            async with asyncio.timeout(_CHAT_TIMEOUT):
                async for event in graph.astream_events(
                    {
                        "messages": initial_messages,
                        "user_id": ctx.user_id,
                        "agent_id": ctx.agent_id,
                        "shared_memory": prepared.resolved_memory.shared_memory,
                    },
                    config=run_config,
                    version="v2",
                ):
                    event_type = event.get("event", "")

                    if event_type == "on_chat_model_start":
                        current_response = ""
                        _llm_start_time = time.monotonic()

                    elif event_type == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk is not None and hasattr(chunk, "content"):
                            raw_content = chunk.content
                            if isinstance(raw_content, str):
                                # Plain string — always final output
                                if raw_content:
                                    current_response += raw_content
                                    yield ChatEvent(type="token", payload={"content": raw_content})
                            elif isinstance(raw_content, list):
                                # Content blocks — separate thinking from output
                                for block in raw_content:
                                    block_type = block.get("type", "") if isinstance(block, dict) else getattr(block, "type", "")
                                    block_text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                                    # Also check "thinking" key (Anthropic format)
                                    if not block_text and isinstance(block, dict):
                                        block_text = block.get("thinking", "")
                                    if block_text:
                                        if block_type == "thinking":
                                            current_thinking += block_text
                                            yield ChatEvent(type="thinking", payload={"content": block_text})
                                        else:
                                            current_response += block_text
                                            yield ChatEvent(type="token", payload={"content": block_text})
                            elif raw_content:
                                text = str(raw_content)
                                current_response += text
                                yield ChatEvent(type="token", payload={"content": text})

                    elif event_type == "on_chat_model_end":
                        latency_ms = int((time.monotonic() - _llm_start_time) * 1000)
                        output = event.get("data", {}).get("output")
                        usage = getattr(output, "usage_metadata", None) or {}
                        # Build full input for LangFuse (system prompt + user message)
                        llm_input = [
                            {"role": "system", "content": prepared.system_prompt},
                            {"role": "user", "content": ctx.message},
                        ]
                        ctx.collector.add_llm_span(
                            model=self._model_name,
                            input=llm_input,
                            output=current_response,
                            prompt_tokens=usage.get("input_tokens", 0),
                            completion_tokens=usage.get("output_tokens", 0),
                            latency_ms=latency_ms,
                        )

                    elif event_type == "on_tool_start":
                        tool_call_count += 1
                        tool_name = event.get("name", "")
                        _tool_start_times[tool_name] = time.monotonic()
                        _pending_tool_args[tool_name] = event["data"].get("input", {})
                        if tool_call_count > _MAX_TOOL_CALLS:
                            ctx.collector.finalize(
                                output=current_response,
                                error=f"工具调用次数超过上限（{_MAX_TOOL_CALLS}次）",
                            )
                            yield ChatEvent(
                                type="error",
                                payload={"message": f"工具调用次数超过上限（{_MAX_TOOL_CALLS}次）。"},
                            )
                            return
                        yield ChatEvent(
                            type="tool_call",
                            payload={"tool": tool_name, "args": event["data"].get("input", {})},
                        )

                    elif event_type == "on_tool_end":
                        tool_name = event.get("name", "")
                        t_start = _tool_start_times.pop(tool_name, time.monotonic())
                        duration_ms = int((time.monotonic() - t_start) * 1000)
                        output = event["data"].get("output", "")
                        output_str = output.content if hasattr(output, "content") else str(output)
                        ctx.collector.add_tool_span(
                            name=tool_name,
                            args=_pending_tool_args.pop(tool_name, {}),
                            result=output_str,
                            latency_ms=duration_ms,
                        )
                        yield ChatEvent(
                            type="tool_result",
                            payload={"tool": tool_name, "result": output_str},
                        )

        except TimeoutError:
            ctx.collector.finalize(output=current_response, error="对话超时（120秒）")
            yield ChatEvent(type="error", payload={"message": "对话超时（120秒），请稍后重试。"})
            return

        # Store response for downstream nodes
        ctx.results["response"] = current_response
        ctx.results["thinking"] = current_thinking or None

    # --- BaseNode abstract methods (not used for stream, but required) ---

    async def _execute(self, ctx: NodeContext) -> NodeResult:
        raise NotImplementedError("Use stream() instead of run() for ReactNode")

    def _report(self, ctx: NodeContext, result: NodeResult, latency_ms: int) -> None:
        pass  # Reporting is done inline during stream()
