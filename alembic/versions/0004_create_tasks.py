"""create tasks table

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create enum types (IF NOT EXISTS avoids errors on re-run with asyncpg)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE task_priority AS ENUM ('low', 'medium', 'high');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE task_status AS ENUM ('todo', 'in_progress', 'done');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE task_source AS ENUM ('manual', 'agent');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)

    op.create_table(
        "tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column(
            "priority",
            sa.Enum("low", "medium", "high", name="task_priority"),
            nullable=False,
            server_default="medium",
        ),
        sa.Column(
            "status",
            sa.Enum("todo", "in_progress", "done", name="task_status"),
            nullable=False,
            server_default="todo",
        ),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column(
            "source",
            sa.Enum("manual", "agent", name="task_source"),
            nullable=False,
            server_default="manual",
        ),
        sa.Column("tags", postgresql.JSONB(), nullable=True),
        sa.Column("related_memory_id", sa.String(200), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index("idx_tasks_user_status_created", "tasks", ["user_id", "status", "created_at"])
    op.create_index("idx_tasks_user_due_date", "tasks", ["user_id", "due_date"])


def downgrade() -> None:
    op.drop_index("idx_tasks_user_due_date", table_name="tasks")
    op.drop_index("idx_tasks_user_status_created", table_name="tasks")
    op.drop_table("tasks")

    op.execute("DROP TYPE IF EXISTS task_source")
    op.execute("DROP TYPE IF EXISTS task_status")
    op.execute("DROP TYPE IF EXISTS task_priority")
