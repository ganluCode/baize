"""Knowledge service layer: KnowledgeBaseService, IngestionService, RetrievalService."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
from baize.knowledge.models import KnowledgeBaseModel, KnowledgeDocumentModel
from baize.knowledge.retrieval.bm25 import BM25Retriever
from baize.knowledge.retrieval.exceptions import KnowledgeBaseNotActiveError
from baize.knowledge.retrieval.expansion import expand_parents
from baize.knowledge.retrieval.hybrid import HybridRetriever
from baize.knowledge.retrieval.types import ScoredChunk
from baize.knowledge.retrieval.vector import VectorRetriever
from baize.knowledge.schemas import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
)
from baize.llm.provider import ModelNotFoundError, ProviderFactory, ProviderNotFoundError, ProviderUnavailableError

if TYPE_CHECKING:
    from baize.core.observability import TraceCollector

logger = logging.getLogger(__name__)


class KnowledgeBaseServiceError(Exception):
    """Raised when a KnowledgeBase service operation fails."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class KnowledgeBaseService:
    """Orchestrates knowledge base business logic.

    Args:
        db: SQLAlchemy async session for database operations.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        data: KnowledgeBaseCreateRequest,
        provider_factory: ProviderFactory,
    ) -> KnowledgeBaseResponse:
        """Create a new knowledge base after validating the embedding provider/model.

        Args:
            user_id: Owner of the new knowledge base.
            data: Validated creation payload.
            provider_factory: LLM provider manager for embedding provider validation.

        Returns:
            The newly created KnowledgeBaseResponse.

        Raises:
            KnowledgeBaseServiceError: 400 if embedding provider/model not found or unavailable.
            KnowledgeBaseServiceError: 409 if a KB with the same name already exists for this user.
        """
        try:
            provider_factory.get_embedding_model(data.embedding_provider, data.embedding_model)
        except (ProviderNotFoundError, ModelNotFoundError, ProviderUnavailableError) as exc:
            raise KnowledgeBaseServiceError(str(exc), status_code=400) from exc

        existing = (
            await self._db.execute(
                select(KnowledgeBaseModel).where(
                    and_(
                        KnowledgeBaseModel.user_id == user_id,
                        KnowledgeBaseModel.name == data.name,
                        KnowledgeBaseModel.status != "deleted",
                    )
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise KnowledgeBaseServiceError(
                f"Knowledge base with name '{data.name}' already exists.",
                status_code=409,
            )

        kb = KnowledgeBaseModel(
            user_id=user_id,
            name=data.name,
            description=data.description,
            embedding_provider=data.embedding_provider,
            embedding_model=data.embedding_model,
            embedding_dim=data.embedding_dim,
            settings=data.settings,
        )
        self._db.add(kb)
        await self._db.commit()
        await self._db.refresh(kb)
        return self._to_response(kb, document_count=0)

    async def list(
        self,
        *,
        user_id: uuid.UUID,
        status: str = "active",
        limit: int = 20,
        offset: int = 0,
    ) -> KnowledgeBaseListResponse:
        """List knowledge bases for a user with pagination.

        Args:
            user_id: Owner to filter by.
            status: KB status filter (default 'active').
            limit: Maximum items to return.
            offset: Number of items to skip.

        Returns:
            KnowledgeBaseListResponse with items and total count.
        """
        count_stmt = (
            select(func.count())
            .select_from(KnowledgeBaseModel)
            .where(
                KnowledgeBaseModel.user_id == user_id,
                KnowledgeBaseModel.status == status,
            )
        )
        total: int = (await self._db.execute(count_stmt)).scalar_one()

        doc_count_subq = (
            select(func.count(KnowledgeDocumentModel.id))
            .where(KnowledgeDocumentModel.kb_id == KnowledgeBaseModel.id)
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )

        stmt = (
            select(KnowledgeBaseModel, doc_count_subq.label("doc_count"))
            .where(
                KnowledgeBaseModel.user_id == user_id,
                KnowledgeBaseModel.status == status,
            )
            .order_by(KnowledgeBaseModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await self._db.execute(stmt)).all()
        items = [self._to_response(kb, int(count)) for kb, count in rows]
        return KnowledgeBaseListResponse(items=items, total=total)

    async def get(self, *, kb_id: uuid.UUID, user_id: uuid.UUID) -> KnowledgeBaseResponse:
        """Retrieve a single knowledge base, enforcing ownership and non-deleted status.

        Args:
            kb_id: Knowledge base to retrieve.
            user_id: The requesting user's id.

        Returns:
            KnowledgeBaseResponse for the matching KB.

        Raises:
            KnowledgeBaseServiceError: 403 if KB exists but belongs to another user.
            KnowledgeBaseServiceError: 404 if KB not found or status is 'deleted'.
        """
        doc_count_subq = (
            select(func.count(KnowledgeDocumentModel.id))
            .where(KnowledgeDocumentModel.kb_id == KnowledgeBaseModel.id)
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )

        stmt = select(KnowledgeBaseModel, doc_count_subq.label("doc_count")).where(
            KnowledgeBaseModel.id == kb_id
        )
        row = (await self._db.execute(stmt)).one_or_none()

        if row is None:
            raise KnowledgeBaseServiceError("Knowledge base not found.", status_code=404)
        kb, doc_count = row
        if kb.user_id != user_id:
            raise KnowledgeBaseServiceError(
                "Access to this knowledge base is forbidden.", status_code=403
            )
        if kb.status == "deleted":
            raise KnowledgeBaseServiceError("Knowledge base not found.", status_code=404)
        return self._to_response(kb, int(doc_count))

    async def soft_delete(self, *, kb_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Soft-delete a knowledge base by setting its status to 'deleted'.

        Args:
            kb_id: Knowledge base to delete.
            user_id: The requesting user's id.

        Raises:
            KnowledgeBaseServiceError: 403 if KB belongs to another user.
            KnowledgeBaseServiceError: 404 if KB not found or already deleted.
        """
        stmt = select(KnowledgeBaseModel).where(KnowledgeBaseModel.id == kb_id)
        kb = (await self._db.execute(stmt)).scalar_one_or_none()

        if kb is None or kb.status == "deleted":
            raise KnowledgeBaseServiceError("Knowledge base not found.", status_code=404)
        if kb.user_id != user_id:
            raise KnowledgeBaseServiceError(
                "Access to this knowledge base is forbidden.", status_code=403
            )

        kb.status = "deleted"
        await self._db.commit()

    def _to_response(self, kb: KnowledgeBaseModel, document_count: int) -> KnowledgeBaseResponse:
        return KnowledgeBaseResponse(
            id=kb.id,
            user_id=kb.user_id,
            name=kb.name,
            description=kb.description,
            embedding_provider=kb.embedding_provider,
            embedding_model=kb.embedding_model,
            embedding_dim=kb.embedding_dim,
            settings=kb.settings,
            status=kb.status,
            document_count=document_count,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        )


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
