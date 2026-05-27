"""Pydantic models for OpenAI-compatible API request/response formats."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str
    # 推理内容（DeepSeek/火山方舟等 OpenAI 兼容协议的扩展字段）
    reasoning_content: str | None = None


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
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None
    # 标准 OpenAI 字段：reasoning_effort（"none" = 关闭思考，其他值 = 开启）
    reasoning_effort: str | None = None
    # Baize 扩展字段（可选，高级客户端可用）
    thinking: bool | None = None
    agent_id: str | None = None
    session_id: str | None = None

    def resolve_thinking(self) -> bool | None:
        """统一为二元思考开关：thinking 字段优先，否则看 reasoning_effort。"""
        if self.thinking is not None:
            return self.thinking
        if self.reasoning_effort is None:
            return None
        return self.reasoning_effort != "none"


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

class ReasoningConfig(BaseModel):
    """Reasoning/thinking configuration."""

    effort: str = "medium"  # none | minimal | low | medium | high


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[OpenAIMessage] | list[dict]
    instructions: str | None = None
    conversation: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    store: bool = True
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    # Baize 扩展
    thinking: bool | None = None
    agent_id: str | None = None

    def resolve_thinking(self) -> bool | None:
        """统一为二元思考开关：thinking 字段优先，否则看 reasoning.effort。"""
        if self.thinking is not None:
            return self.thinking
        if self.reasoning is None:
            return None
        return self.reasoning.effort != "none"


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
