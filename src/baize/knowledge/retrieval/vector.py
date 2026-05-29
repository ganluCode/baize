"""Vector similarity retriever backed by Qdrant."""

import json
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
from baize.knowledge.retrieval.exceptions import VectorRetrievalError
from baize.knowledge.retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)

_HYDRATE_SQL = text(
    """
    SELECT
        id,
        doc_id,
        kb_id,
        content,
        section_path,
        level,
        parent_chunk_id,
        qdrant_point_id
    FROM knowledge_chunks
    WHERE qdrant_point_id = ANY(:point_ids)
    """
)


def _is_collection_not_found(exc: Exception) -> bool:
    """Return True if exc indicates a missing Qdrant collection."""
    # REST transport: qdrant_client raises UnexpectedResponse with HTTP 404
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse  # noqa: PLC0415

        if isinstance(exc, UnexpectedResponse) and exc.status_code == 404:
            return True
    except ImportError:
        pass
    # In-memory / gRPC fallback: check error message
    msg = str(exc).lower()
    return "not found" in msg and "collection" in msg


class VectorRetriever:
    """Retrieves knowledge chunks using Qdrant vector similarity search.

    Args:
        db_session: SQLAlchemy async session for hydrating chunks from PostgreSQL.
        qdrant_client: Async Qdrant client for vector search.
        embedder: KnowledgeEmbedder for encoding the query.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        qdrant_client: AsyncQdrantClient,
        embedder: KnowledgeEmbedder,
    ) -> None:
        self._session = db_session
        self._qdrant = qdrant_client
        self._embedder = embedder

    async def search(
        self,
        *,
        kb_id: uuid.UUID,
        query: str,
        limit: int,
    ) -> list[ScoredChunk]:
        """Search chunks by cosine vector similarity.

        Args:
            kb_id: Knowledge base to search within.
            query: Natural language search query.
            limit: Maximum number of results to return.

        Returns:
            List of ScoredChunk ordered by Qdrant cosine similarity descending.
            Returns [] when the collection does not exist.

        Raises:
            VectorRetrievalError: When Qdrant is unreachable or returns an unexpected error.
        """
        vector = await self._embedder.embed_query(query)
        collection_name = f"baize_kb_{kb_id.hex}"

        try:
            qdrant_results = await self._qdrant.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
            )
        except Exception as exc:
            if _is_collection_not_found(exc):
                logger.debug("Collection '%s' not found; returning empty list", collection_name)
                return []
            raise VectorRetrievalError(
                f"Qdrant search failed for collection '{collection_name}': {exc}"
            ) from exc

        if not qdrant_results:
            return []

        # Build ordered point_id list and score mapping
        ordered_point_ids: list[uuid.UUID] = []
        point_id_to_score: dict[uuid.UUID, float] = {}
        for scored_point in qdrant_results:
            pid = uuid.UUID(str(scored_point.id))
            ordered_point_ids.append(pid)
            point_id_to_score[pid] = float(scored_point.score)

        # Batch-hydrate from PostgreSQL (asyncpg converts Python list → pg array)
        db_result = await self._session.execute(
            _HYDRATE_SQL,
            {"point_ids": ordered_point_ids},
        )
        rows = db_result.mappings().all()
        rows_by_point_id: dict[uuid.UUID, Any] = {
            row["qdrant_point_id"]: row for row in rows
        }

        # Re-assemble in Qdrant order
        chunks: list[ScoredChunk] = []
        for rank, point_id in enumerate(ordered_point_ids, start=1):
            row = rows_by_point_id.get(point_id)
            if row is None:
                logger.warning("Qdrant point_id=%s has no matching chunk in PostgreSQL", point_id)
                continue

            raw_path = row["section_path"]
            if raw_path:
                try:
                    section_path: list[str] = json.loads(raw_path)
                except (json.JSONDecodeError, ValueError):
                    section_path = [raw_path]
            else:
                section_path = []

            score = point_id_to_score[point_id]
            chunks.append(
                ScoredChunk(
                    chunk_id=row["id"],
                    doc_id=row["doc_id"],
                    kb_id=row["kb_id"],
                    content=row["content"],
                    section_path=section_path,
                    level=row["level"],
                    parent_chunk_id=row["parent_chunk_id"],
                    score=score,
                    rank=rank,
                    sources=["vector"],
                    bm25_rank=None,
                    vector_rank=rank,
                    bm25_score=None,
                    vector_score=score,
                )
            )

        return chunks
