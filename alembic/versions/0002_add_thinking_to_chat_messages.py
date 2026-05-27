"""Add thinking column to chat_messages.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-14
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("thinking", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "thinking")
