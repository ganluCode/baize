"""create sessions and chat_messages tables

Revision ID: 0001
Revises: 0000
Create Date: 2026-03-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ENUM as PGEnum

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = '0000'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create a stub agent_configs table so the FK from sessions can be enforced.
    op.create_table(
        "agent_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
    )

    # Create enum types using pg_type check (asyncpg-safe)
    conn = op.get_bind()
    for type_name, values in [
        ("session_status", ["active", "archived"]),
        ("message_role", ["user", "assistant", "system", "tool"]),
    ]:
        exists = conn.execute(
            sa.text("SELECT 1 FROM pg_type WHERE typname = :name"), {"name": type_name}
        ).scalar()
        if not exists:
            vals = ", ".join(f"\'{v}\'" for v in values)
            conn.execute(sa.text(f"CREATE TYPE {type_name} AS ENUM ({vals})"))

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent_configs.id"), nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("auto_memory_recall", sa.Boolean(), nullable=True),
        sa.Column("shared_memory", sa.Boolean(), nullable=True),
        sa.Column("status", PGEnum(name="session_status", create_type=False), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sessions_user_agent_updated", "sessions", ["user_id", "agent_id", "updated_at"])

    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", PGEnum(name="message_role", create_type=False), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        sa.Column("tool_call_id", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_chat_messages_session_created", "chat_messages", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_session_created", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_sessions_user_agent_updated", table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("agent_configs")
    op.execute("DROP TYPE message_role")
    op.execute("DROP TYPE session_status")
