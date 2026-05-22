"""SQLAlchemy async ORM model for agent_configs table."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from baize.user.models import Base


class AgentConfig(Base):
    """ORM model mapping to the agent_configs table.

    字段设计说明:
    - prompts: 分层 prompt 配置
      {"soul": "人设/身份", "behavior": "任务/行为指令"}
      soul 和 behavior 都可选；未来可扩展 examples / constraints 等层
    - tools: 分层工具配置
      {"builtin": ["save_memory"], "mcp_servers": ["web-search"], "skills": ["summarize_url"]}
    - model_config: LLM 路由配置
      {"chat": "openai/gpt-4o", "reasoning": "anthropic/claude-opus-4-6"}
    - memory_config: 记忆策略（替代原 auto_memory_recall + shared_memory）
      {"auto_recall": true, "shared": true, "top_k": 5}
    - sub_agents: 可委派的子智能体 ID 列表
      ["uuid-1", "uuid-2"]
    - guardrails: 执行护栏
      {"max_tool_calls": 10, "timeout_seconds": 120, "max_tokens_per_turn": 4096}
    """

    __tablename__ = "agent_configs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, default="chat", server_default="chat")
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")

    # 核心配置
    prompts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_config_json: Mapped[dict | None] = mapped_column("model_config", JSONB, nullable=True)
    tools: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    sub_agents: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # 记忆策略（合并原 auto_memory_recall + shared_memory）
    memory_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # 执行护栏
    guardrails: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("ix_agent_configs_user_enabled", "user_id", "is_enabled"),)
