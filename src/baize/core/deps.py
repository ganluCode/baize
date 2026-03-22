"""FastAPI dependency injection bridge for Baize services.

Provides :func:`get_container` and per-service factory functions that route
FastAPI ``Depends`` calls to the global :class:`~baize.core.container.Container`
instance initialised in the application lifespan.
"""

from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.repository import AgentConfigRepository
from baize.agent.service import AgentConfigService, AgentService
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
    """Dependency factory for TaskService.

    TODO: Return typed TaskService once the task feature is implemented.
    """
    return container.task_service


def get_provider_factory(container: Container = Depends(get_container)) -> ProviderFactory:
    """Dependency factory for ProviderFactory."""
    return container.provider_factory


def get_model_router(container: Container = Depends(get_container)) -> ModelRouter:
    """Dependency factory for ModelRouter."""
    return container.model_router


def get_agent_service(
    db: AsyncSession = Depends(get_db),
    model_router: ModelRouter = Depends(get_model_router),
    memory_service: MemoryServiceInterface | None = Depends(get_memory_service),
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
    )
