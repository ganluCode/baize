"""OpenAI Chat Completions compatible endpoint.

POST /v1/chat/completions — stateless chat (messages array carries full history).
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
from baize.core.deps import get_agent_service
from baize.openai_compat.helpers import resolve_agent_by_model
from baize.openai_compat.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    OpenAIMessage,
)
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai-compat"])


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_svc: AgentService = Depends(get_agent_service),
):
    """OpenAI Chat Completions compatible endpoint (stateless).

    The client sends the full message history in ``messages``.
    Baize injects system prompt and memory but does NOT load session history.
    """
    uid = current_user.id

    # 1. Resolve agent
    agent = await resolve_agent_by_model(body.model, uid, db)

    # 2. Extract current message (last user message)
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message found")
    current_message = user_msgs[-1].content

    # 3. Create a transient session for tracking (or use provided session_id)
    session_id = uuid.UUID(body.session_id) if body.session_id else uuid.uuid4()

    # 4. Call AgentService.chat()
    # NOTE: In stateless mode the client already sent full history in messages.
    # AgentService will load session history from DB (which is empty for transient sessions),
    # and ContextService will assemble prompt + recall memory normally.
    # The client-provided history is used by the LLM via the messages array.
    chat_iter = agent_svc.chat(
        agent_id=agent.id,
        session_id=session_id,
        user_id=uid,
        message=current_message,
        user=current_user,
    )

    if body.stream:
        return EventSourceResponse(_completions_sse_stream(chat_iter, body.model))
    else:
        return await _completions_collect(chat_iter, body.model)


async def _completions_sse_stream(
    chat_iter: AsyncIterator[ChatEvent],
    model: str,
) -> AsyncIterator[dict]:
    """Convert Baize ChatEvent stream to Chat Completions SSE format."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    try:
        async for event in chat_iter:
            if event.type == "token":
                chunk = ChatCompletionResponse(
                    id=chat_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        ChatCompletionChoice(
                            delta={"content": event.payload.get("content", "")},
                        )
                    ],
                )
                yield {"data": chunk.model_dump_json()}
            elif event.type == "done":
                # Final chunk with finish_reason
                chunk = ChatCompletionResponse(
                    id=chat_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        ChatCompletionChoice(delta={}, finish_reason="stop")
                    ],
                )
                yield {"data": chunk.model_dump_json()}
            elif event.type == "error":
                logger.error("Chat completions stream error: %s", event.payload)
                break
    except Exception as exc:
        logger.error("Chat completions stream exception: %s", exc, exc_info=True)

    yield {"data": "[DONE]"}


async def _completions_collect(
    chat_iter: AsyncIterator[ChatEvent],
    model: str,
) -> ChatCompletionResponse:
    """Collect all ChatEvents into a non-streaming Chat Completions response."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    content_parts: list[str] = []

    async for event in chat_iter:
        if event.type == "token":
            content_parts.append(event.payload.get("content", ""))
        elif event.type == "error":
            raise HTTPException(
                status_code=500,
                detail=event.payload.get("message", "Agent error"),
            )

    return ChatCompletionResponse(
        id=chat_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                message=OpenAIMessage(role="assistant", content="".join(content_parts)),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(),
    )
