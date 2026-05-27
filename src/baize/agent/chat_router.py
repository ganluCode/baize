"""Chat endpoints: POST /api/v1/chat (SSE streaming, F-009) and /api/v1/chat/sync (F-010)."""

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.service import AgentConfigService, AgentService, AgentServiceError, ChatEvent
from baize.agent.streaming import create_sse_response
from baize.auth.deps import get_current_user
from baize.core.database import get_db
from baize.core.deps import get_agent_config_service, get_agent_service, get_session_service
from baize.core.schemas import success
from baize.session.service import SessionService
from baize.user.models import UserModel
from baize.user.repository import UserRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatStreamRequest(BaseModel):
    """Request body for POST /api/v1/chat."""

    message: str
    agent_id: uuid.UUID | None = None
    session_id: uuid.UUID | None = None
    # 深度思考开关：true=开启，false=关闭，省略=用模型默认
    thinking: bool | None = None


async def _get_user_model(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserModel:
    """Load the full UserModel from the authenticated user_id string."""
    repo = UserRepository(db)
    user = await repo.get_by_id(uuid.UUID(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("")
async def chat_stream(
    body: ChatStreamRequest,
    user: UserModel = Depends(_get_user_model),
    agent_svc: AgentService = Depends(get_agent_service),
    agent_config_svc: AgentConfigService = Depends(get_agent_config_service),
    session_svc: SessionService = Depends(get_session_service),
):
    """Stream a chat turn as server-sent events (SSE).

    Resolves the agent (default if not specified), validates session ownership,
    auto-creates a session if needed, then delegates to AgentService.chat().

    Returns:
        EventSourceResponse with Content-Type: text/event-stream.

    Raises:
        HTTPException: 400 if message is empty, 403 if session not owned,
            404 if default agent not found.
    """
    uid = user.id

    # 1. Validate message
    if not body.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 2. Resolve agent
    if body.agent_id is not None:
        try:
            agent = await agent_config_svc.get(body.agent_id, uid)
        except AgentServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    else:
        agent = await agent_config_svc.get_default(uid)
        if agent is None:
            raise HTTPException(status_code=404, detail="No default agent found.")

    # 3. Validate session ownership (if session_id provided)
    if body.session_id is not None:
        await session_svc.get_session(body.session_id, uid)

    # 4. Resolve session_id for AgentService (use existing or let it auto-create)
    effective_session_id = body.session_id if body.session_id is not None else uuid.uuid4()

    # 5. Stream chat events with error wrapping
    chat_iter = agent_svc.chat(
        agent_id=agent.id,
        session_id=effective_session_id,
        user_id=uid,
        message=body.message,
        user=user,
        thinking=body.thinking,
    )
    return create_sse_response(_safe_chat_stream(chat_iter))


class ChatSyncResponse(BaseModel):
    """Response body for POST /api/v1/chat/sync."""

    session_id: str
    message_id: str
    content: str
    tool_calls: list[dict[str, Any]]


@router.post("/sync")
async def chat_sync(
    body: ChatStreamRequest,
    user: UserModel = Depends(_get_user_model),
    agent_svc: AgentService = Depends(get_agent_service),
    agent_config_svc: AgentConfigService = Depends(get_agent_config_service),
    session_svc: SessionService = Depends(get_session_service),
):
    """Return a complete chat turn as a single JSON response.

    Collects all events from AgentService.chat() and assembles the full
    response before returning.

    Returns:
        ApiResponse wrapping ChatSyncResponse with code=0.

    Raises:
        HTTPException: 400 if message is empty, 403 if session not owned,
            404 if default agent not found, 500 on agent error.
    """
    uid = user.id

    # 1. Validate message
    if not body.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 2. Resolve agent
    if body.agent_id is not None:
        try:
            agent = await agent_config_svc.get(body.agent_id, uid)
        except AgentServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    else:
        agent = await agent_config_svc.get_default(uid)
        if agent is None:
            raise HTTPException(status_code=404, detail="No default agent found.")

    # 3. Validate session ownership (if session_id provided)
    if body.session_id is not None:
        await session_svc.get_session(body.session_id, uid)

    # 4. Resolve session_id for AgentService
    effective_session_id = body.session_id if body.session_id is not None else uuid.uuid4()

    # 5. Collect all chat events
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    session_id_result: str = str(effective_session_id)
    message_id_result: str = ""

    chat_iter = agent_svc.chat(
        agent_id=agent.id,
        session_id=effective_session_id,
        user_id=uid,
        message=body.message,
        user=user,
        thinking=body.thinking,
    )

    async for event in chat_iter:
        if event.type == "token":
            content_parts.append(event.payload.get("content", ""))
        elif event.type == "tool_call":
            pending_tool_calls.append({
                "tool": event.payload.get("tool", ""),
                "args": event.payload.get("args", {}),
                "result": "",
            })
        elif event.type == "tool_result":
            tool_name = event.payload.get("tool", "")
            # Find the last pending tool_call with this tool name and attach result
            for tc in reversed(pending_tool_calls):
                if tc["tool"] == tool_name and tc["result"] == "":
                    tc["result"] = event.payload.get("result", "")
                    tool_calls.append(tc)
                    pending_tool_calls.remove(tc)
                    break
        elif event.type == "done":
            session_id_result = event.payload.get("session_id", session_id_result)
            message_id_result = event.payload.get("message_id", "")
        elif event.type == "error":
            raise HTTPException(
                status_code=500,
                detail=event.payload.get("message", "Agent error"),
            )

    return success(
        ChatSyncResponse(
            session_id=session_id_result,
            message_id=message_id_result,
            content="".join(content_parts),
            tool_calls=tool_calls,
        )
    )


async def _safe_chat_stream(
    chat_iter: AsyncIterator[ChatEvent],
) -> AsyncIterator[ChatEvent]:
    """Wrap a chat iterator to catch exceptions and emit an error event."""
    try:
        async for event in chat_iter:
            yield event
    except Exception as exc:
        logger.error("Agent error during chat stream: %s", exc, exc_info=True)
        yield ChatEvent(type="error", payload={"message": str(exc)})
