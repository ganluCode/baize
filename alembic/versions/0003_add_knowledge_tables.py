"""Add knowledge module tables: knowledge_bases, knowledge_documents, knowledge_chunks.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-28

Tables: knowledge_bases, knowledge_documents, knowledge_chunks
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    # ── knowledge_bases ───────────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL,
            name VARCHAR(100) NOT NULL,
            description VARCHAR(500),
            embedding_provider VARCHAR(50) NOT NULL,
            embedding_model VARCHAR(100) NOT NULL,
            embedding_dim INTEGER NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            settings JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_bases_user_status"
        " ON knowledge_bases (user_id, status)"
    ))

    # ── knowledge_documents ───────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id UUID PRIMARY KEY,
            kb_id UUID NOT NULL,
            title VARCHAR(255) NOT NULL,
            source_type VARCHAR(20) NOT NULL,
            source_uri VARCHAR(1024),
            content_hash VARCHAR(64) NOT NULL,
            doc_metadata JSONB,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            error_message VARCHAR(2000),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_documents_kb_status"
        " ON knowledge_documents (kb_id, status)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_documents_kb_hash"
        " ON knowledge_documents (kb_id, content_hash)"
    ))

    # ── knowledge_chunks ──────────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id UUID PRIMARY KEY,
            kb_id UUID NOT NULL,
            doc_id UUID NOT NULL,
            parent_chunk_id UUID,
            level INTEGER NOT NULL,
            section_path VARCHAR(500),
            content TEXT NOT NULL,
            content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
            chunk_metadata JSONB,
            qdrant_point_id UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_doc"
        " ON knowledge_chunks (doc_id)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_kb_level"
        " ON knowledge_chunks (kb_id, level)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_content_tsv"
        " ON knowledge_chunks USING gin (content_tsv)"
    ))


def downgrade() -> None:
    op.drop_table("knowledge_chunks")
    op.drop_table("knowledge_documents")
    op.drop_table("knowledge_bases")
