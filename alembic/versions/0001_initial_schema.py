"""Initial schema — all tables from scratch.

Revision ID: 0001
Revises: None
Create Date: 2026-04-04

Tables: system_users, agent_configs, sessions, chat_messages, tasks
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── system_users ──────────────────────────────────────────────
    op.create_table(
        "system_users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("password", sa.Text(), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="user"),
        sa.Column("avatar", sa.Text(), nullable=True),
        sa.Column("api_key_hash", sa.Text(), nullable=True),
        sa.Column("preferences", postgresql.JSONB(), nullable=True),
        sa.Column("default_agent_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # ── agent_configs ─────────────────────────────────────────────
    op.create_table(
        "agent_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("agent_type", sa.String(50), nullable=False, server_default="chat"),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("prompts", postgresql.JSONB(), nullable=True),
        sa.Column("model_config", postgresql.JSONB(), nullable=True),
        sa.Column("tools", postgresql.JSONB(), nullable=True),
        sa.Column("sub_agents", postgresql.JSONB(), nullable=True),
        sa.Column("memory_config", postgresql.JSONB(), nullable=True),
        sa.Column("guardrails", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_agent_configs_user_enabled", "agent_configs", ["user_id", "is_enabled"])

    # ── sessions ──────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("title_gen_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Enum("active", "archived", name="session_status"), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sessions_user_agent_updated", "sessions", ["user_id", "agent_id", "updated_at"])

    # ── chat_messages ─────────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.Enum("user", "assistant", "system", "tool", name="message_role"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        sa.Column("tool_name", sa.String(100), nullable=True),
        sa.Column("token_usage", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_chat_messages_session_created", "chat_messages", ["session_id", "created_at"])

    # ── tasks ─────────────────────────────────────────────────────
    op.create_table(
        "tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("priority", sa.Enum("low", "medium", "high", name="task_priority"), nullable=False, server_default="medium"),
        sa.Column("status", sa.Enum("todo", "in_progress", "done", name="task_status"), nullable=False, server_default="todo"),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("source", sa.Enum("manual", "agent", name="task_source"), nullable=False, server_default="manual"),
        sa.Column("tags", postgresql.JSONB(), nullable=True),
        sa.Column("related_memory_id", sa.String(200), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("idx_tasks_user_status_created", "tasks", ["user_id", "status", "created_at"])
    op.create_index("idx_tasks_user_due_date", "tasks", ["user_id", "due_date"])


def downgrade() -> None:
    op.drop_table("tasks")
    op.drop_table("chat_messages")
    op.drop_table("sessions")
    op.drop_table("agent_configs")
    op.drop_table("system_users")
    op.execute("DROP TYPE IF EXISTS session_status")
    op.execute("DROP TYPE IF EXISTS message_role")
    op.execute("DROP TYPE IF EXISTS task_priority")
    op.execute("DROP TYPE IF EXISTS task_status")
    op.execute("DROP TYPE IF EXISTS task_source")
