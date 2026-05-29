"""FastAPI dependency injection for the knowledge module."""

from collections.abc import Callable

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from baize.core.database import get_db
from baize.core.deps import get_provider_factory
from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
from baize.knowledge.ingestion.qdrant_client import get_qdrant_client
from baize.knowledge.ingestion.service import IngestionService
from baize.knowledge.models import KnowledgeBaseModel
from baize.knowledge.service import KnowledgeBaseService, RetrievalService
from baize.llm.provider import ProviderFactory


def get_knowledge_base_service(
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBaseService:
    """Dependency factory for KnowledgeBaseService — builds a per-request instance."""
    return KnowledgeBaseService(db)


async def get_ingestion_service(
    db: AsyncSession = Depends(get_db),
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> Callable[[KnowledgeBaseModel], IngestionService]:
    """Dependency factory for IngestionService.

    Returns a factory callable that accepts a KnowledgeBaseModel and creates an
    IngestionService configured for that knowledge base's embedding settings.
    The Qdrant client is resolved from the shared singleton.
    """
    qdrant = await get_qdrant_client()

    def make_ingestion_service(kb: KnowledgeBaseModel) -> IngestionService:
        embedder = KnowledgeEmbedder(
            provider_name=kb.embedding_provider,
            model_id=kb.embedding_model,
            expected_dim=kb.embedding_dim,
            llm_provider_manager=provider_factory,
        )
        return IngestionService(session=db, embedder=embedder, qdrant=qdrant)

    return make_ingestion_service


async def get_retrieval_service(
    db: AsyncSession = Depends(get_db),
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> RetrievalService:
    """Dependency factory for RetrievalService — builds a per-request instance."""
    qdrant = await get_qdrant_client()

    def make_embedder(kb: KnowledgeBaseModel) -> KnowledgeEmbedder:
        return KnowledgeEmbedder(
            provider_name=kb.embedding_provider,
            model_id=kb.embedding_model,
            expected_dim=kb.embedding_dim,
            llm_provider_manager=provider_factory,
        )

    return RetrievalService(
        db_session=db,
        qdrant_client=qdrant,
        embedder_factory=make_embedder,
    )
