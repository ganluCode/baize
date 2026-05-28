"""OpenAI Responses API compatible endpoints.

POST /v1/responses         — stateful chat (maps to AgentService.chat)
POST /v1/conversations     — create a conversation (maps to session)
GET  /v1/conversations/{id} — get conversation info
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from baize.agent.service import AgentService, ChatEvent
from baize.core.database import get_db
from baize.core.deps import get_agent_service, get_session_service
from baize.openai_compat.helpers import resolve_agent_by_model
from baize.openai_compat.schemas import (
    ConversationCreateRequest,
    ConversationObject,
    ConversationRef,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
)
from baize.session.service import SessionService
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)


def _extract_text_from_content(content) -> str:
    """Extract plain text from various content formats.

    Handles:
    - str: "hello" → "hello"
    - list of content blocks: [{"type": "input_text", "text": "hello"}] → "hello"
    - list of content blocks: [{"type": "text", "text": "hello"}] → "hello"
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
    return str(content) if content else ""


def _iter_input_items(input_data):
    """Yield (role, content_text) pairs from Responses API input."""
    if not isinstance(input_data, list):
        return
    for item in input_data:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
        elif hasattr(item, "role"):
            role = item.role
            content = item.content if hasattr(item, "content") else ""
        else:
            continue
        yield role, _extract_text_from_content(content)


def _extract_message(input_data) -> str:
    """Extract the latest user message from Responses API input."""
    if isinstance(input_data, str):
        return input_data
    if not isinstance(input_data, list) or not input_data:
        return ""

    user_texts = [text for role, text in _iter_input_items(input_data) if role == "user"]
    if user_texts:
        return user_texts[-1]

    # Fallback: take last item's content regardless of role
    last = input_data[-1]
    content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    return _extract_text_from_content(content)


def _extract_history(input_data):
    """Extract all messages BEFORE the last user message as LangChain history.

    Returns None when input is a string (single message) or empty list.
    System messages are skipped (Baize injects its own).
    """
    if not isinstance(input_data, list) or len(input_data) <= 1:
        return None

    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    items = list(_iter_input_items(input_data))
    # Find index of the last user message — everything before it is history
    last_user_idx = next(
        (i for i in range(len(items) - 1, -1, -1) if items[i][0] == "user"),
        None,
    )
    if last_user_idx is None or last_user_idx == 0:
        return None

    history: list[BaseMessage] = []
    for role, text in items[:last_user_idx]:
        if role == "user":
            history.append(HumanMessage(content=text))
        elif role == "assistant":
            history.append(AIMessage(content=text))
        # system / tool / others — skipped
    return history or None


router = APIRouter(tags=["openai-compat"])


# ---------------------------------------------------------------------------
# POST /v1/responses
# ---------------------------------------------------------------------------


@router.post("/responses")
async def create_response(
    body: ResponsesRequest,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_svc: AgentService = Depends(get_agent_service),
    session_svc: SessionService = Depends(get_session_service),
):
    """Create a response (OpenAI Responses API compatible).

    Maps ``model`` to a Baize agent and ``conversation`` to a session.
    Supports both streaming and non-streaming modes.
    """
    uid = current_user.id
    logger.info("Responses API request: model=%s, input_type=%s, conversation=%s, stream=%s",
                body.model, type(body.input).__name__, body.conversation, body.stream)

    # 1. Resolve agent
    agent = await resolve_agent_by_model(body.model, uid, db)

    # 2. Resolve conversation → session
    if body.conversation:
        try:
            session_id = uuid.UUID(body.conversation)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation id")
    else:
        # Auto-create session
        session = await session_svc.create_session(uid, agent.id)
        session_id = session.id

    # 3. Extract current message + prior history from input
    message = _extract_message(body.input)
    if not message:
        raise HTTPException(status_code=400, detail="input cannot be empty")
    # If input is a list with prior messages, use them as history (overrides DB lookup)
    external_history = _extract_history(body.input)

    # 4. Resolve thinking toggle (binary on/off)
    thinking = body.resolve_thinking()

    # 5. Call AgentService.chat()
    chat_iter = agent_svc.chat(
        agent_id=agent.id,
        session_id=session_id,
        user_id=uid,
        message=message,
        user=current_user,
        thinking=thinking,
        external_history=external_history,
    )

    if body.stream:
        return EventSourceResponse(_responses_sse_stream(chat_iter, session_id, body.model))
    else:
        return await _responses_collect(chat_iter, session_id, body.model)


async def _responses_sse_stream(
    chat_iter: AsyncIterator[ChatEvent],
    session_id: uuid.UUID,
    model: str,
) -> AsyncIterator[dict]:
    """Convert Baize ChatEvent stream to Responses API SSE events."""
    response_id = f"resp-{uuid.uuid4().hex[:24]}"
    now = int(time.time())

    # response.created
    yield {
        "event": "response.created",
        "data": json.dumps({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": now,
                "model": model,
                "status": "in_progress",
                "output": [],
                "conversation": {"id": str(session_id)},
            },
        }, ensure_ascii=False),
    }

    try:
        async for event in chat_iter:
            if event.type == "token":
                yield {
                    "event": "response.output_text.delta",
                    "data": json.dumps({
                        "type": "response.output_text.delta",
                        "delta": event.payload.get("content", ""),
                    }, ensure_ascii=False),
                }
            elif event.type == "thinking":
                yield {
                    "event": "response.thinking.delta",
                    "data": json.dumps({
                        "type": "response.thinking.delta",
                        "delta": event.payload.get("content", ""),
                    }, ensure_ascii=False),
                }
            elif event.type == "tool_call":
                yield {
                    "event": "response.function_call_arguments.delta",
                    "data": json.dumps({
                        "type": "response.function_call_arguments.delta",
                        "name": event.payload.get("tool", ""),
                        "delta": json.dumps(event.payload.get("args", {}), ensure_ascii=False),
                    }, ensure_ascii=False),
                }
            elif event.type == "tool_result":
                yield {
                    "event": "response.function_call_output.done",
                    "data": json.dumps({
                        "type": "response.function_call_output.done",
                        "name": event.payload.get("tool", ""),
                        "output": event.payload.get("result", ""),
                    }, ensure_ascii=False),
                }
            elif event.type == "error":
                yield {
                    "event": "response.failed",
                    "data": json.dumps({
                        "type": "response.failed",
                        "error": {"message": event.payload.get("message", "")},
                    }, ensure_ascii=False),
                }
                return
    except Exception as exc:
        logger.error("Responses stream error: %s", exc, exc_info=True)
        yield {
            "event": "response.failed",
            "data": json.dumps({
                "type": "response.failed",
                "error": {"message": str(exc)},
            }, ensure_ascii=False),
        }
        return

    # response.completed
    yield {
        "event": "response.completed",
        "data": json.dumps({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": now,
                "model": model,
                "status": "completed",
                "conversation": {"id": str(session_id)},
            },
        }, ensure_ascii=False),
    }


async def _responses_collect(
    chat_iter: AsyncIterator[ChatEvent],
    session_id: uuid.UUID,
    model: str,
) -> ResponseObject:
    """Collect all ChatEvents into a non-streaming Responses API response."""
    response_id = f"resp-{uuid.uuid4().hex[:24]}"
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    message_id = ""

    async for event in chat_iter:
        if event.type == "token":
            content_parts.append(event.payload.get("content", ""))
        elif event.type == "thinking":
            thinking_parts.append(event.payload.get("content", ""))
        elif event.type == "done":
            message_id = event.payload.get("message_id", "")
        elif event.type == "error":
            raise HTTPException(
                status_code=500,
                detail=event.payload.get("message", "Agent error"),
            )

    output: list = []
    if thinking_parts:
        from baize.openai_compat.schemas import ResponseOutputReasoning, ResponseReasoningSummary
        output.append(
            ResponseOutputReasoning(
                id=f"rs-{uuid.uuid4().hex[:16]}",
                summary=[ResponseReasoningSummary(text="".join(thinking_parts))],
            )
        )
    output.append(
        ResponseOutputMessage(
            id=message_id or f"msg-{uuid.uuid4().hex[:16]}",
            content=[ResponseOutputText(text="".join(content_parts))],
        )
    )

    return ResponseObject(
        id=response_id,
        created_at=int(time.time()),
        model=model,
        output=output,
        conversation=ConversationRef(id=str(session_id)),
    )


# ---------------------------------------------------------------------------
# POST /v1/conversations
# ---------------------------------------------------------------------------


@router.post("/conversations", response_model=ConversationObject)
async def create_conversation(
    body: ConversationCreateRequest | None = None,
    current_user: UserModel = Depends(get_current_user),
    session_svc: SessionService = Depends(get_session_service),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation (maps to a Baize session).

    The conversation is created without an agent binding — the first
    ``POST /v1/responses`` call with this conversation will bind it.
    """
    # Get default agent for the user to create a session
    from baize.user.models import UserModel as UM
    from sqlalchemy import select

    user_result = await db.execute(
        select(UM.default_agent_id).where(UM.id == current_user.id)
    )
    default_agent_id = user_result.scalar_one_or_none()
    if default_agent_id is None:
        raise HTTPException(status_code=404, detail="No default agent configured")

    session = await session_svc.create_session(current_user.id, default_agent_id)

    return ConversationObject(
        id=str(session.id),
        created_at=int(session.created_at.timestamp()) if session.created_at else 0,
    )


# ---------------------------------------------------------------------------
# GET /v1/conversations/{conversation_id}
# ---------------------------------------------------------------------------


@router.get("/conversations/{conversation_id}", response_model=ConversationObject)
async def get_conversation(
    conversation_id: str,
    current_user: UserModel = Depends(get_current_user),
    session_svc: SessionService = Depends(get_session_service),
):
    """Get conversation info (maps to a Baize session)."""
    try:
        session_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation id")

    session = await session_svc.get_session(session_uuid, current_user.id)
    return ConversationObject(
        id=str(session.id),
        created_at=int(session.created_at.timestamp()) if session.created_at else 0,
    )
