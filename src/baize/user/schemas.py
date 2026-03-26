"""Pydantic request/response schemas for the user module."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PreferencesSchema(BaseModel):
    """User preferences configuration."""

    communication_style: str = "friendly"
    language: str = "zh"
    focus_areas: list[str] = []
    custom_instructions: str | None = None
    auto_memory_recall: bool = True
    shared_memory: bool = False


class UserResponse(BaseModel):
    """Public user representation — excludes password and api_key_hash."""

    id: uuid.UUID
    email: str
    name: str
    role: str
    avatar: str | None
    preferences: dict[str, Any] | None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserCreateRequest(BaseModel):
    """Request body for creating a new user."""

    email: str
    name: str
    password: str
    role: str = "user"
    avatar: str | None = None
    preferences: dict[str, Any] | None = None


class UserCreateResponse(UserResponse):
    """Response for user creation — includes one-time plaintext api_key."""

    api_key: str


class UserUpdateRequest(BaseModel):
    """Request body for updating the current user."""

    name: str | None = None
    avatar: str | None = None
    preferences: dict[str, Any] | None = None


class UserListResponse(BaseModel):
    """Paginated list of users."""

    items: list[UserResponse]
    total: int


class ResetKeyResponse(BaseModel):
    """Response for API key reset — contains new plaintext key."""

    api_key: str
