"""Pydantic request/response schemas for the agent module."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentCreate(BaseModel):
    """Request body for creating a new agent configuration."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=100)
    system_prompt: str
    tools: list[str]
    description: str | None = None
    # 'model_config' is reserved by Pydantic v2; use alias to expose the JSON key
    llm_config: dict[str, Any] | None = Field(default=None, alias="model_config")
    auto_memory_recall: bool | None = None
    shared_memory: bool | None = None
    is_default: bool = False


class AgentUpdate(BaseModel):
    """Request body for partially updating an agent configuration (PATCH semantics)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(default=None, min_length=1, max_length=100)
    system_prompt: str | None = None
    tools: list[str] | None = None
    description: str | None = None
    llm_config: dict[str, Any] | None = Field(default=None, alias="model_config")
    auto_memory_recall: bool | None = None
    shared_memory: bool | None = None
    is_default: bool | None = None


class ChatRequest(BaseModel):
    """Request body for the chat SSE endpoint."""

    message: str = Field(..., min_length=1)


class AgentResponse(BaseModel):
    """Agent configuration representation returned by the API."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    description: str | None
    system_prompt: str
    tools: list[str] | None
    llm_config: dict[str, Any] | None = Field(default=None, alias="model_config")
    auto_memory_recall: bool | None
    shared_memory: bool | None
    is_default: bool
    created_at: datetime
    updated_at: datetime
