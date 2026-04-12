"""Pydantic models for OpenAI-compatible API request/response formats."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class OpenAIMessage(BaseModel):
    role: str
    content: str


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "baize"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None
    # Baize 扩展字段（可选，高级客户端可用）
    agent_id: str | None = None
    session_id: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None


# ---------------------------------------------------------------------------
# POST /v1/responses
# ---------------------------------------------------------------------------

class ResponsesRequest(BaseModel):
    model: str
    input: str | list[OpenAIMessage]
    instructions: str | None = None
    conversation: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    store: bool = True
    temperature: float | None = None
    max_output_tokens: int | None = None
    # Baize 扩展
    agent_id: str | None = None


class ResponseOutputText(BaseModel):
    type: str = "output_text"
    text: str = ""
    annotations: list[Any] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    type: str = "message"
    id: str = ""
    role: str = "assistant"
    content: list[ResponseOutputText] = Field(default_factory=list)


class ConversationRef(BaseModel):
    id: str


class ResponseObject(BaseModel):
    id: str
    object: str = "response"
    created_at: int = 0
    model: str = ""
    output: list[ResponseOutputMessage] = Field(default_factory=list)
    conversation: ConversationRef | None = None
    status: str = "completed"


# ---------------------------------------------------------------------------
# POST /v1/conversations
# ---------------------------------------------------------------------------

class ConversationCreateRequest(BaseModel):
    """Empty body — just creates a new conversation."""
    pass


class ConversationObject(BaseModel):
    id: str
    object: str = "conversation"
    created_at: int = 0
