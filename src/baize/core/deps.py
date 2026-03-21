"""FastAPI dependency injection bridge for Baize services.

Provides :func:`get_container` and per-service factory functions that route
FastAPI ``Depends`` calls to the global :class:`~baize.core.container.Container`
instance initialised in the application lifespan.
"""

from typing import Any

from fastapi import Depends

from baize.core.container import Container
from baize.llm.model_router import ModelRouter
from baize.llm.provider import ProviderFactory

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


def get_memory_service(container: Container = Depends(get_container)) -> Any:
    """Dependency factory for MemoryService.

    TODO: Return typed MemoryService once the memory feature is implemented.
    """
    return container.memory_service


def get_session_service(container: Container = Depends(get_container)) -> Any:
    """Dependency factory for SessionService.

    TODO: Return typed SessionService once the session feature is implemented.
    """
    return container.session_service


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


def get_agent_service(container: Container = Depends(get_container)) -> Any:
    """Dependency factory for AgentService.

    TODO: Return typed AgentService once the agent feature is implemented.
    """
    return container.agent_service


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
