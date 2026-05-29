"""FastAPI dependency injection bridge for Baize services.

Provides :func:`get_container` and per-service factory functions that route
FastAPI ``Depends`` calls to the global :class:`~baize.core.container.Container`
instance initialised in the application lifespan.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.repository import AgentConfigRepository
from baize.agent.service import AgentConfigService, AgentService
from baize.context.service import ContextService
from baize.core.container import Container
from baize.core.database import get_db
from baize.llm.model_router import ModelRouter
from baize.llm.provider import ProviderFactory
from baize.memory.interface import MemoryServiceInterface
from baize.session.repository import ChatMessageRepository, SessionRepository
from baize.session.service import SessionService

# Global container instance — set by main.py lifespan via set_container().
_container: Container | None = None


def set_container(container: Container) -> None:
    """Store the global container instance (called from lifespan on startup)."""
    global _container
    _container = container


def get_container() -> Container:
    """Return the global :class:`Container` instance.

    Raises:
        RuntimeError: If the container has not been initialised yet.
    """
    if _container is None:
        raise RuntimeError("Container is not initialised. Ensure lifespan has run.")
    return _container


# ---------------------------------------------------------------------------
# Per-service dependency factories
# Each function is a FastAPI-compatible dependency that extracts a service
# from the container.  The service may be None while the owning feature is
# not yet implemented — callers should guard accordingly.
# ---------------------------------------------------------------------------


def get_memory_service(
    container: Container = Depends(get_container),
) -> MemoryServiceInterface | None:
    """Dependency factory for MemoryServiceInterface (Mem0Adapter when configured)."""
    return container.memory_service


def get_session_service(db: AsyncSession = Depends(get_db)) -> SessionService:
    """Dependency factory for SessionService — builds a per-request instance."""
    session_repo = SessionRepository(db)
    message_repo = ChatMessageRepository(db)
    return SessionService(session_repo=session_repo, message_repo=message_repo)


def get_user_service(container: Container = Depends(get_container)) -> Any:
    """Dependency factory for UserService.

    TODO: Return typed UserService once the user feature is implemented.
    """
    return container.user_service


def get_auth_service(container: Container = Depends(get_container)) -> Any:
    """Dependency factory for AuthService.

    TODO: Return typed AuthService once the auth feature is implemented.
    """
    return container.auth_service


def get_agent_config_service(db: AsyncSession = Depends(get_db)) -> AgentConfigService:
    """Dependency factory for AgentConfigService — builds a per-request instance."""
    agent_repo = AgentConfigRepository(db)
    session_repo = SessionRepository(db)
    message_repo = ChatMessageRepository(db)
    session_svc = SessionService(session_repo=session_repo, message_repo=message_repo)
    return AgentConfigService(repository=agent_repo, session_service=session_svc)


def get_task_service(container: Container = Depends(get_container)) -> Any:
    """Dependency factory for TaskEngineService (used by agent tools)."""
    return container.task_service


def get_provider_factory(container: Container = Depends(get_container)) -> ProviderFactory:
    """Dependency factory for ProviderFactory."""
    return container.provider_factory


def get_model_router(container: Container = Depends(get_container)) -> ModelRouter:
    """Dependency factory for ModelRouter."""
    return container.model_router


def get_context_service(
    db: AsyncSession = Depends(get_db),
    memory_service: MemoryServiceInterface | None = Depends(get_memory_service),
    container: Container = Depends(get_container),
) -> ContextService:
    """Dependency factory for ContextService — builds a per-request instance."""
    session_repo = SessionRepository(db)
    message_repo = ChatMessageRepository(db)
    session_svc = SessionService(session_repo=session_repo, message_repo=message_repo)
    return ContextService(
        session_service=session_svc,
        memory_service=memory_service,
        settings=container.config,
    )


def get_retrieval_svc_factory(
    db: AsyncSession = Depends(get_db),
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> Callable[[], Awaitable[Any]]:
    """Return a lazy factory that creates RetrievalService on first call.

    Qdrant is only connected when the factory is invoked (i.e. for knowledge
    agents), so chat-only requests have zero knowledge-layer overhead.
    """
    async def make() -> Any:
        from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
        from baize.knowledge.ingestion.qdrant_client import get_qdrant_client
        from baize.knowledge.models import KnowledgeBaseModel
        from baize.knowledge.service import RetrievalService

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

    return make


def get_agent_service(
    db: AsyncSession = Depends(get_db),
    model_router: ModelRouter = Depends(get_model_router),
    memory_service: MemoryServiceInterface | None = Depends(get_memory_service),
    context_service: ContextService = Depends(get_context_service),
    retrieval_svc_factory: Callable[[], Awaitable[Any]] = Depends(get_retrieval_svc_factory),
) -> AgentService:
    """Dependency factory for AgentService — builds a per-request instance."""
    agent_repo = AgentConfigRepository(db)
    session_repo = SessionRepository(db)
    message_repo = ChatMessageRepository(db)
    session_svc = SessionService(session_repo=session_repo, message_repo=message_repo)
    agent_config_svc = AgentConfigService(repository=agent_repo, session_service=session_svc)
    return AgentService(
        agent_config_service=agent_config_svc,
        session_service=session_svc,
        memory_service=memory_service,
        model_router=model_router,
        context_service=context_service,
        retrieval_svc_provider=retrieval_svc_factory,
    )
