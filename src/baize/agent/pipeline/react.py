"""Graph execution step — runs the LangGraph for the agent's type and streams ChatEvents.

Despite the historical name "ReactStep", this step is **graph-agnostic**: it
looks up the graph builder by ``agent_config.agent_type`` (chat / rag / ...)
and runs whatever graph that builder produces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from baize.agent.graphs import get_graph_builder
from baize.agent.pipeline.base import BaseStep, StepContext, StepResult
from baize.agent.service import ChatEvent
from baize.agent.tools import ToolRegistry

if TYPE_CHECKING:
    from baize.agent.models import AgentConfig
    from baize.context.service import PreparedContext

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 10
_CHAT_TIMEOUT = 120.0

# LangChain message type → OpenAI-style role
_LC_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}


def _lc_message_role(msg) -> str:
    """Map a LangChain message instance to an OpenAI-style role string."""
    msg_type = getattr(msg, "type", None) or msg.__class__.__name__.lower().replace("message", "")
    return _LC_TYPE_TO_ROLE.get(msg_type, msg_type or "user")


def _lc_message_content(msg) -> str:
    """Extract text content from a LangChain message (handles list[ContentBlock])."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("thinking") or "")
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
    return str(content) if content else ""


class ReactStep(BaseStep):
    """Executes the LangGraph for the agent_type, streaming ChatEvents.

    Unlike other steps, this is a generator — call ``stream()`` instead of
    ``run()`` to get ChatEvents. Tracing is still automatic.
    """

    step_name = "react"

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

    async def stream(self, ctx: StepContext) -> AsyncIterator[ChatEvent]:
        """Stream ChatEvents from the agent's graph with automatic tracing."""
        prepared: "PreparedContext" = ctx.results["prepared"]

        # Resolve tools
        tool_names: list[str] = []
        if self._agent_config.tools and isinstance(self._agent_config.tools, dict):
            tool_names.extend(self._agent_config.tools.get("builtin", []))
        elif self._agent_config.tools and isinstance(self._agent_config.tools, list):
            tool_names.extend(self._agent_config.tools)
        tool_entries = ToolRegistry.get_by_names(tool_names)
        tools = [e.langchain_tool for e in tool_entries if e.langchain_tool is not None]

        # Build graph via registry — builder applies thinking internally per its own logic
        agent_type = getattr(self._agent_config, "agent_type", "chat") or "chat"
        builder = get_graph_builder(agent_type)
        graph = builder.build(
            llm=self._llm,
            tools=tools,
            agent_config=self._agent_config,
            thinking=self._thinking,
        )

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
        _llm_input_messages: list[dict] = []  # captured from on_chat_model_start
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
                        # Capture the actual messages LangGraph is sending to the LLM
                        # (includes system + full history + current user message + tool exchanges)
                        _input = event.get("data", {}).get("input", {})
                        msgs = _input.get("messages") if isinstance(_input, dict) else None
                        if msgs and isinstance(msgs, list) and isinstance(msgs[0], list):
                            msgs = msgs[0]  # LangChain may wrap as [[msg, msg, ...]]
                        _llm_input_messages = [
                            {
                                "role": _lc_message_role(m),
                                "content": _lc_message_content(m),
                            }
                            for m in (msgs or [])
                        ]

                    elif event_type == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk is not None and hasattr(chunk, "content"):
                            raw_content = chunk.content
                            if isinstance(raw_content, str):
                                if raw_content:
                                    current_response += raw_content
                                    yield ChatEvent(type="token", payload={"content": raw_content})
                            elif isinstance(raw_content, list):
                                for block in raw_content:
                                    block_type = block.get("type", "") if isinstance(block, dict) else getattr(block, "type", "")
                                    block_text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
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
                        ctx.collector.add_llm_span(
                            model=self._model_name,
                            input=_llm_input_messages,
                            output=current_response,
                            thinking=current_thinking,
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

        # Store response for downstream steps
        ctx.results["response"] = current_response
        ctx.results["thinking"] = current_thinking or None

    # --- BaseStep abstract methods (not used for stream, but required) ---

    async def _execute(self, ctx: StepContext) -> StepResult:
        raise NotImplementedError("Use stream() instead of run() for ReactStep")

    def _report(self, ctx: StepContext, result: StepResult, latency_ms: int) -> None:
        pass  # Reporting is done inline during stream()
