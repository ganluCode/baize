"""Knowledge service layer: KnowledgeBaseService, IngestionService, RetrievalService."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
from baize.knowledge.models import KnowledgeBaseModel
from baize.knowledge.retrieval.bm25 import BM25Retriever
from baize.knowledge.retrieval.exceptions import KnowledgeBaseNotActiveError
from baize.knowledge.retrieval.expansion import expand_parents
from baize.knowledge.retrieval.hybrid import HybridRetriever
from baize.knowledge.retrieval.types import ScoredChunk
from baize.knowledge.retrieval.vector import VectorRetriever

if TYPE_CHECKING:
    from baize.core.observability import TraceCollector

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Orchestrates knowledge base business logic.

    TODO: Implement methods for knowledge base CRUD, document ingestion,
    and retrieval coordination.
    """

    pass


class RetrievalService:
    """Executes hybrid BM25 + vector retrieval over a knowledge base.

    Args:
        db_session: SQLAlchemy async session.
        qdrant_client: Async Qdrant client for vector search.
        embedder_factory: Callable that takes a KnowledgeBaseModel and returns a
            KnowledgeEmbedder configured for that KB's embedding settings.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        qdrant_client: AsyncQdrantClient,
        embedder_factory: Callable[[KnowledgeBaseModel], KnowledgeEmbedder],
    ) -> None:
        self._session = db_session
        self._qdrant = qdrant_client
        self._embedder_factory = embedder_factory

    async def search(
        self,
        *,
        kb_id: uuid.UUID,
        query: str,
        top_k: int | None = None,
        include_parents: bool = False,
        tracer: TraceCollector | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve relevant chunks from a knowledge base using hybrid search.

        Args:
            kb_id: Knowledge base to search within.
            query: Natural language search query.
            top_k: Maximum results to return; defaults to settings.knowledge_default_top_k.
            include_parents: When True, appends parent chunks to the result list.
            tracer: Optional trace collector for observability (reserved, not yet used).

        Returns:
            List of ScoredChunk sorted by RRF score descending.

        Raises:
            KnowledgeBaseNotActiveError: If the KB does not exist or its status != 'active'.
            VectorDimMismatchError: If the embedder's output dimension mismatches the KB config.
            VectorRetrievalError: If Qdrant is unreachable.
        """
        start = time.monotonic()

        kb = await self._session.get(KnowledgeBaseModel, kb_id)
        if kb is None or kb.status != "active":
            status = kb.status if kb is not None else "not_found"
            raise KnowledgeBaseNotActiveError(
                f"Knowledge base {kb_id} is not searchable (status={status})"
            )

        # May raise VectorDimMismatchError — caller gets it unmodified
        embedder = self._embedder_factory(kb)

        from baize.core.config import settings

        effective_top_k = top_k if top_k is not None else settings.knowledge_default_top_k

        bm25 = BM25Retriever(db_session=self._session)
        vector = VectorRetriever(
            db_session=self._session,
            qdrant_client=self._qdrant,
            embedder=embedder,
        )
        hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

        results = await hybrid.search(
            kb_id=kb_id,
            query=query,
            top_k=effective_top_k,
            candidate_limit=effective_top_k * 4,
        )

        if include_parents:
            results = await expand_parents(db_session=self._session, chunks=results)

        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "RetrievalService.search kb_id=%s query='%s' hits=%d latency_ms=%.1f",
            str(kb_id),
            query[:80],
            len(results),
            latency_ms,
        )

        return results
