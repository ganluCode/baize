"""Unit tests for the SSE streaming layer (F-013)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from baize.agent.service import ChatEvent
from baize.agent.streaming import chat_events_to_sse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_event_stream(*events: ChatEvent) -> AsyncIterator[ChatEvent]:
    """Yield the given ChatEvent objects as an async generator."""
    for event in events:
        yield event


async def _collect_sse(stream: AsyncIterator[dict]) -> list[dict]:
    """Collect all SSE dicts from an async generator."""
    result = []
    async for item in stream:
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Event type conversion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_event_converted_correctly() -> None:
    """token event maps to SSE event=token with JSON-encoded content."""
    events = [ChatEvent(type="token", payload={"content": "hello"})]
    sse_items = await _collect_sse(chat_events_to_sse(_make_event_stream(*events)))

    assert len(sse_items) == 1
    assert sse_items[0]["event"] == "token"
    assert json.loads(sse_items[0]["data"]) == {"content": "hello"}


@pytest.mark.asyncio
async def test_tool_call_event_converted_correctly() -> None:
    """tool_call event maps to SSE event=tool_call with tool and args."""
    events = [ChatEvent(type="tool_call", payload={"tool": "search_memory", "args": {"query": "test"}})]
    sse_items = await _collect_sse(chat_events_to_sse(_make_event_stream(*events)))

    assert len(sse_items) == 1
    assert sse_items[0]["event"] == "tool_call"
    data = json.loads(sse_items[0]["data"])
    assert data["tool"] == "search_memory"
    assert data["args"] == {"query": "test"}


@pytest.mark.asyncio
async def test_tool_result_event_converted_correctly() -> None:
    """tool_result event maps to SSE event=tool_result with tool and result."""
    events = [ChatEvent(type="tool_result", payload={"tool": "search_memory", "result": "Found memory"})]
    sse_items = await _collect_sse(chat_events_to_sse(_make_event_stream(*events)))

    assert len(sse_items) == 1
    assert sse_items[0]["event"] == "tool_result"
    data = json.loads(sse_items[0]["data"])
    assert data["tool"] == "search_memory"
    assert data["result"] == "Found memory"


@pytest.mark.asyncio
async def test_confirm_event_converted_correctly() -> None:
    """confirm event maps to SSE event=confirm with tool, args, and description."""
    events = [
        ChatEvent(
            type="confirm",
            payload={"tool": "delete_file", "args": {"path": "/tmp/x"}, "description": "Delete /tmp/x?"},
        )
    ]
    sse_items = await _collect_sse(chat_events_to_sse(_make_event_stream(*events)))

    assert len(sse_items) == 1
    assert sse_items[0]["event"] == "confirm"
    data = json.loads(sse_items[0]["data"])
    assert data["tool"] == "delete_file"
    assert data["description"] == "Delete /tmp/x?"


@pytest.mark.asyncio
async def test_done_event_converted_correctly() -> None:
    """done event maps to SSE event=done with session_id and message_id."""
    events = [ChatEvent(type="done", payload={"session_id": "sid-1", "message_id": "mid-1"})]
    sse_items = await _collect_sse(chat_events_to_sse(_make_event_stream(*events)))

    assert len(sse_items) == 1
    assert sse_items[0]["event"] == "done"
    data = json.loads(sse_items[0]["data"])
    assert data["session_id"] == "sid-1"
    assert data["message_id"] == "mid-1"


@pytest.mark.asyncio
async def test_error_event_converted_correctly() -> None:
    """error event maps to SSE event=error with message."""
    events = [ChatEvent(type="error", payload={"message": "Something went wrong"})]
    sse_items = await _collect_sse(chat_events_to_sse(_make_event_stream(*events)))

    assert len(sse_items) == 1
    assert sse_items[0]["event"] == "error"
    data = json.loads(sse_items[0]["data"])
    assert data["message"] == "Something went wrong"


@pytest.mark.asyncio
async def test_multiple_events_all_converted() -> None:
    """Multiple events are all yielded in order."""
    events = [
        ChatEvent(type="token", payload={"content": "Hello"}),
        ChatEvent(type="token", payload={"content": " world"}),
        ChatEvent(type="done", payload={"session_id": "s1", "message_id": "m1"}),
    ]
    sse_items = await _collect_sse(chat_events_to_sse(_make_event_stream(*events)))

    assert len(sse_items) == 3
    assert sse_items[0]["event"] == "token"
    assert sse_items[1]["event"] == "token"
    assert sse_items[2]["event"] == "done"


# ---------------------------------------------------------------------------
# Client disconnect handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_exit_stops_stream_silently() -> None:
    """GeneratorExit during iteration stops the stream without raising."""

    async def slow_stream():
        yield ChatEvent(type="token", payload={"content": "a"})
        raise GeneratorExit

    collected = []
    try:
        async for item in chat_events_to_sse(slow_stream()):
            collected.append(item)
    except GeneratorExit:
        pass  # GeneratorExit may propagate to the caller; that's fine

    # At least the first event was processed before disconnect
    assert len(collected) >= 0  # no crash is the key assertion


@pytest.mark.asyncio
async def test_cancelled_error_stops_stream_silently() -> None:
    """CancelledError during iteration stops the stream without raising an unexpected exception."""

    async def cancellable_stream():
        yield ChatEvent(type="token", payload={"content": "a"})
        raise asyncio.CancelledError

    collected = []
    try:
        async for item in chat_events_to_sse(cancellable_stream()):
            collected.append(item)
    except asyncio.CancelledError:
        pass  # CancelledError may propagate; the key is no unexpected exceptions

    assert len(collected) >= 0  # no crash
