"""BM25 full-text retriever backed by PostgreSQL ts_rank."""

import json
import logging
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from baize.knowledge.retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)

_MAX_QUERY_LEN = 500

_SEARCH_SQL = text(
    """
    SELECT
        id,
        doc_id,
        kb_id,
        content,
        section_path,
        level,
        parent_chunk_id,
        ts_rank(content_tsv, plainto_tsquery('simple', :q)) AS bm25_score
    FROM knowledge_chunks
    WHERE kb_id = :kb_id
      AND content_tsv @@ plainto_tsquery('simple', :q)
    ORDER BY bm25_score DESC
    LIMIT :limit
    """
)


class BM25Retriever:
    """Retrieves knowledge chunks using PostgreSQL full-text search.

    Args:
        db_session: SQLAlchemy async session.
    """

    def __init__(self, db_session: AsyncSession) -> None:
        self._session = db_session

    async def search(
        self,
        *,
        kb_id: uuid.UUID,
        query: str,
        limit: int,
    ) -> list[ScoredChunk]:
        """Search chunks by BM25 (ts_rank) full-text similarity.

        Args:
            kb_id: Knowledge base to search within.
            query: Natural language search query.
            limit: Maximum number of results to return.

        Returns:
            List of ScoredChunk ordered by ts_rank descending, rank is 1-based.
            Returns [] immediately when query is empty or whitespace.
        """
        if not query or not query.strip():
            return []

        q = query[:_MAX_QUERY_LEN]

        result = await self._session.execute(
            _SEARCH_SQL,
            {"kb_id": kb_id, "q": q, "limit": limit},
        )
        rows = result.mappings().all()

        chunks: list[ScoredChunk] = []
        for rank, row in enumerate(rows, start=1):
            raw_path = row["section_path"]
            if raw_path:
                try:
                    section_path: list[str] = json.loads(raw_path)
                except (json.JSONDecodeError, ValueError):
                    section_path = [raw_path]
            else:
                section_path = []

            bm25_score = float(row["bm25_score"])
            chunks.append(
                ScoredChunk(
                    chunk_id=row["id"],
                    doc_id=row["doc_id"],
                    kb_id=row["kb_id"],
                    content=row["content"],
                    section_path=section_path,
                    level=row["level"],
                    parent_chunk_id=row["parent_chunk_id"],
                    score=bm25_score,
                    rank=rank,
                    sources=["bm25"],
                    bm25_rank=rank,
                    vector_rank=None,
                    bm25_score=bm25_score,
                    vector_score=None,
                )
            )

        return chunks
