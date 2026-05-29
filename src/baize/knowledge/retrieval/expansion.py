"""Parent chunk expansion for retrieval results."""

import json
import logging
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from baize.knowledge.retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)

_FETCH_BY_IDS_SQL = text(
    """
    SELECT
        id,
        doc_id,
        kb_id,
        content,
        section_path,
        level,
        parent_chunk_id
    FROM knowledge_chunks
    WHERE id = ANY(:chunk_ids)
    """
)


async def expand_parents(
    *,
    db_session: AsyncSession,
    chunks: list[ScoredChunk],
    include_parent_levels: int = 1,
) -> list[ScoredChunk]:
    """Append parent chunks to retrieval results.

    For each chunk that has a parent_chunk_id, fetches the parent from PostgreSQL
    and appends it to the result list. Chunks at level=0 or with no parent are
    skipped. Already-present parents are not duplicated. No recursive expansion.

    Args:
        db_session: SQLAlchemy async session.
        chunks: Original retrieval results (order preserved in output).
        include_parent_levels: Reserved for future multi-level support; currently
                               only one level of expansion is performed.

    Returns:
        Original chunks followed by fetched parent chunks.
        Parent chunks have score=0.0, sources=["parent_expansion"], and
        rank starting from len(original_chunks)+1.
    """
    if not chunks:
        return []

    existing_ids: set[uuid.UUID] = {c.chunk_id for c in chunks}

    # Collect unique parent IDs that are not already in the input list
    parent_ids_ordered: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for chunk in chunks:
        if chunk.parent_chunk_id is None or chunk.level == 0:
            continue
        pid = chunk.parent_chunk_id
        if pid not in existing_ids and pid not in seen:
            parent_ids_ordered.append(pid)
            seen.add(pid)

    if not parent_ids_ordered:
        return list(chunks)

    # Single batch fetch
    db_result = await db_session.execute(
        _FETCH_BY_IDS_SQL,
        {"chunk_ids": parent_ids_ordered},
    )
    rows = db_result.mappings().all()
    rows_by_id = {row["id"]: row for row in rows}

    result = list(chunks)
    original_len = len(chunks)

    for i, parent_id in enumerate(parent_ids_ordered, start=1):
        row = rows_by_id.get(parent_id)
        if row is None:
            logger.warning("Parent chunk_id=%s not found in PostgreSQL", parent_id)
            continue

        raw_path = row["section_path"]
        if raw_path:
            try:
                section_path: list[str] = json.loads(raw_path)
            except (json.JSONDecodeError, ValueError):
                section_path = [raw_path]
        else:
            section_path = []

        result.append(
            ScoredChunk(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                kb_id=row["kb_id"],
                content=row["content"],
                section_path=section_path,
                level=row["level"],
                parent_chunk_id=row["parent_chunk_id"],
                score=0.0,
                rank=original_len + i,
                sources=["parent_expansion"],
                bm25_rank=None,
                vector_rank=None,
                bm25_score=None,
                vector_score=None,
            )
        )

    return result
