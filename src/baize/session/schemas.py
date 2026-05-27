"""Pydantic request/response schemas for the session module."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from baize.session.models import MessageRole, SessionStatus


class SessionCreate(BaseModel):
    """Request body for creating a new session."""

    title: str | None = Field(default=None, max_length=200)


class SessionUpdate(BaseModel):
    """Request body for updating a session (title and/or status)."""

    title: str | None = Field(default=None, max_length=200)
    status: Literal["active", "archived"] | None = None


class SessionResponse(BaseModel):
    """Session representation returned by the API."""

    id: uuid.UUID
    user_id: uuid.UUID
    agent_id: uuid.UUID
    title: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    """Paginated list of sessions."""

    items: list[SessionResponse]
    total: int


class ChatMessageResponse(BaseModel):
    """Chat message representation returned by the API."""

    id: uuid.UUID
    session_id: uuid.UUID
    user_id: uuid.UUID
    role: MessageRole
    content: str
    thinking: str | None = None
    tool_calls: dict[str, Any] | None
    tool_name: str | None
    token_usage: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageListResponse(BaseModel):
    """Paginated list of chat messages."""

    items: list[ChatMessageResponse]
    total: int
