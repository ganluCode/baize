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

    # 3. Extract current message from input
    if isinstance(body.input, str):
        message = body.input
    elif isinstance(body.input, list) and body.input:
        # Take the last user message
        user_msgs = [m for m in body.input if m.role == "user"]
        message = user_msgs[-1].content if user_msgs else body.input[-1].content
    else:
        raise HTTPException(status_code=400, detail="input cannot be empty")

    # 4. Call AgentService.chat()
    chat_iter = agent_svc.chat(
        agent_id=agent.id,
        session_id=session_id,
        user_id=uid,
        message=message,
        user=current_user,
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
    message_id = ""

    async for event in chat_iter:
        if event.type == "token":
            content_parts.append(event.payload.get("content", ""))
        elif event.type == "done":
            message_id = event.payload.get("message_id", "")
        elif event.type == "error":
            raise HTTPException(
                status_code=500,
                detail=event.payload.get("message", "Agent error"),
            )

    return ResponseObject(
        id=response_id,
        created_at=int(time.time()),
        model=model,
        output=[
            ResponseOutputMessage(
                id=message_id or f"msg-{uuid.uuid4().hex[:16]}",
                content=[ResponseOutputText(text="".join(content_parts))],
            )
        ],
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
