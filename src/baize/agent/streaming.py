"""SSE streaming layer for AgentService.chat() events.

Converts a stream of :class:`~baize.agent.service.ChatEvent` objects into
server-sent events (SSE) via ``sse-starlette``.

Supported event types
---------------------
- ``token``        – LLM output fragment; data: ``{"content": "..."}``
- ``tool_call``    – tool invocation start; data: ``{"tool": "...", "args": {...}}``
- ``tool_result``  – tool invocation result; data: ``{"tool": "...", "result": "..."}``
- ``confirm``      – confirm-level tool requires user approval;
                     data: ``{"tool": "...", "args": {...}, "description": "..."}``
- ``done``         – conversation finished; data: ``{"session_id": "...", "message_id": "..."}``
- ``error``        – fatal error, connection will close; data: ``{"message": "..."}``
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from baize.agent.service import ChatEvent

logger = logging.getLogger(__name__)


async def chat_events_to_sse(
    chat_iter: AsyncIterator[ChatEvent],
) -> AsyncIterator[dict]:
    """Convert a :class:`ChatEvent` async iterator to sse-starlette-compatible dicts.

    Each dict contains ``event`` (the event type string) and ``data`` (a
    JSON-serialised payload string).  The generator catches
    :exc:`GeneratorExit` and :exc:`asyncio.CancelledError` so that a client
    disconnect does not propagate an unexpected exception to the caller.

    Args:
        chat_iter: Async iterator of :class:`ChatEvent` from
            :meth:`~baize.agent.service.AgentService.chat`.

    Yields:
        Dicts accepted by :class:`sse_starlette.sse.EventSourceResponse`.
    """
    try:
        async for event in chat_iter:
            yield {
                "event": event.type,
                "data": json.dumps(event.payload, ensure_ascii=False),
            }
    except (GeneratorExit, asyncio.CancelledError):
        logger.debug("SSE client disconnected; stopping event stream.")
        return


def create_sse_response(chat_iter: AsyncIterator[ChatEvent]) -> EventSourceResponse:
    """Wrap a :class:`ChatEvent` async iterator in an :class:`EventSourceResponse`.

    Args:
        chat_iter: Async iterator produced by
            :meth:`~baize.agent.service.AgentService.chat`.

    Returns:
        An :class:`sse_starlette.sse.EventSourceResponse` suitable for
        returning directly from a FastAPI route handler.
    """
    return EventSourceResponse(chat_events_to_sse(chat_iter))
