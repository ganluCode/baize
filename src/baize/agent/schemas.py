"""Pydantic request/response schemas for the agent module."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentPrompts(BaseModel):
    """分层 prompt 配置。所有层都可选，按存在的层依次组装。"""

    soul: str | None = None      # 人设 / 身份层（"我是谁"）
    behavior: str | None = None  # 任务 / 行为指令层（"我现在要做什么"）


class ToolsConfig(BaseModel):
    """工具配置结构。"""

    builtin: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    """记忆策略配置。"""

    auto_recall: bool = True
    shared: bool = True
    top_k: int = 5


class GuardrailsConfig(BaseModel):
    """执行护栏配置。"""

    max_tool_calls: int = 10
    timeout_seconds: int = 120
    max_tokens_per_turn: int | None = None


class AgentCreate(BaseModel):
    """Request body for creating a new agent configuration."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    agent_type: str = "chat"
    prompts: AgentPrompts = Field(default_factory=AgentPrompts)
    # 'model_config' is reserved by Pydantic v2; use alias to expose the JSON key
    llm_config: dict[str, Any] | None = Field(default=None, alias="model_config")
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    sub_agents: list[str] | None = None
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    set_as_default: bool = False


class AgentUpdate(BaseModel):
    """Request body for partially updating an agent configuration (PATCH semantics)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    agent_type: str | None = None
    prompts: AgentPrompts | None = None
    llm_config: dict[str, Any] | None = Field(default=None, alias="model_config")
    tools: ToolsConfig | None = None
    sub_agents: list[str] | None = None
    memory_config: MemoryConfig | None = None
    guardrails: GuardrailsConfig | None = None
    set_as_default: bool | None = None
    is_enabled: bool | None = None


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
    agent_type: str
    prompts: AgentPrompts | None
    llm_config: dict[str, Any] | None = Field(default=None, alias="model_config", validation_alias="model_config_json")
    tools: ToolsConfig | None
    sub_agents: list[str] | None
    memory_config: MemoryConfig | None
    guardrails: GuardrailsConfig | None
    is_enabled: bool
    created_at: datetime
    updated_at: datetime
