"""expand agent_configs table with all required columns

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add FK constraint and all missing columns to the existing agent_configs stub table.
    op.add_column("agent_configs", sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False))
    op.create_foreign_key(
        "fk_agent_configs_user_id",
        "agent_configs",
        "system_users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.add_column("agent_configs", sa.Column("name", sa.String(100), nullable=False))
    op.add_column("agent_configs", sa.Column("description", sa.String(500), nullable=True))
    op.add_column("agent_configs", sa.Column("system_prompt", sa.Text(), nullable=False))
    op.add_column("agent_configs", sa.Column("tools", postgresql.JSONB(), nullable=True))
    op.add_column("agent_configs", sa.Column("model_config", postgresql.JSONB(), nullable=True))
    op.add_column("agent_configs", sa.Column("auto_memory_recall", sa.Boolean(), nullable=True))
    op.add_column("agent_configs", sa.Column("shared_memory", sa.Boolean(), nullable=True))
    op.add_column(
        "agent_configs",
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "agent_configs",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column(
        "agent_configs",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_agent_configs_user_is_default",
        "agent_configs",
        ["user_id", "is_default"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_configs_user_is_default", table_name="agent_configs")
    op.drop_constraint("fk_agent_configs_user_id", "agent_configs", type_="foreignkey")
    op.drop_column("agent_configs", "updated_at")
    op.drop_column("agent_configs", "created_at")
    op.drop_column("agent_configs", "is_default")
    op.drop_column("agent_configs", "shared_memory")
    op.drop_column("agent_configs", "auto_memory_recall")
    op.drop_column("agent_configs", "model_config")
    op.drop_column("agent_configs", "tools")
    op.drop_column("agent_configs", "system_prompt")
    op.drop_column("agent_configs", "description")
    op.drop_column("agent_configs", "name")
    op.drop_column("agent_configs", "user_id")
