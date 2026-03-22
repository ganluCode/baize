"""create tasks table

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_enum_if_not_exists(conn, name: str, values: list[str]) -> None:
    exists = conn.execute(
        text("SELECT 1 FROM pg_type WHERE typname = :name"), {"name": name}
    ).scalar()
    if not exists:
        vals = ", ".join(f"'{v}'" for v in values)
        conn.execute(text(f"CREATE TYPE {name} AS ENUM ({vals})"))


def upgrade() -> None:
    conn = op.get_bind()
    _create_enum_if_not_exists(conn, "task_priority", ["low", "medium", "high"])
    _create_enum_if_not_exists(conn, "task_status", ["todo", "in_progress", "done"])
    _create_enum_if_not_exists(conn, "task_source", ["manual", "agent"])

    op.create_table(
        "tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column(
            "priority",
            sa.Enum("low", "medium", "high", name="task_priority", create_type=False),
            nullable=False,
            server_default="medium",
        ),
        sa.Column(
            "status",
            sa.Enum("todo", "in_progress", "done", name="task_status", create_type=False),
            nullable=False,
            server_default="todo",
        ),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column(
            "source",
            sa.Enum("manual", "agent", name="task_source", create_type=False),
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
