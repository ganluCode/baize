"""add user_id to chat_messages

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-22

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    # Backfill from parent session
    op.execute("""
        UPDATE chat_messages cm
        SET user_id = s.user_id
        FROM sessions s
        WHERE cm.session_id = s.id
    """)
    op.alter_column("chat_messages", "user_id", nullable=False)


def downgrade() -> None:
    op.drop_column("chat_messages", "user_id")
