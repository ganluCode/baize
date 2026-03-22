"""Application-wide service container for Baize (manual dependency injection)."""

import logging
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine

from baize.core.config import Settings
from baize.llm.model_router import ModelRouter
from baize.llm.provider import ProviderFactory
from baize.llm.schemas import AgentConfig, LLMSettings
from baize.memory.interface import MemoryServiceInterface
from baize.user.jwt_service import JWTService

logger = logging.getLogger(__name__)


class Container:
    """Holds all application services and infrastructure clients.

    Instantiated once at startup via the FastAPI lifespan handler.
    Services are populated by their respective feature implementations;
    they start as None until the owning feature is wired in.
    """

    def __init__(self, config: Settings) -> None:
        # Import engine lazily to avoid triggering Settings() at module import time.
        from baize.core.database import engine as _db_engine

        self.config: Settings = config

        # Infrastructure
        self.db: AsyncEngine = _db_engine
        self.redis: aioredis.Redis = aioredis.from_url(config.redis_url)

        # User / auth
        self.jwt_service: JWTService = JWTService(secret_key=config.secret_key, redis=self.redis)

        # LLM layer
        llm_raw: dict[str, Any] = config.yaml_config.get("llm") or {}
        agents_raw: list[Any] = config.yaml_config.get("agents") or []
        llm_settings = LLMSettings(providers=llm_raw.get("providers", []))
        agents: list[AgentConfig] = [AgentConfig.model_validate(a) for a in agents_raw]
        self.provider_factory: ProviderFactory = ProviderFactory(llm_settings)
        self.model_router: ModelRouter = ModelRouter(self.provider_factory, agents)

        # Memory layer
        self.memory_service: MemoryServiceInterface | None = self._init_memory_service(config)


        # TODO(session): Filled by session feature implementation
        self.session_service: Any | None = None

        # TODO(user): Filled by user feature implementation
        self.user_service: Any | None = None

        # TODO(auth): Filled by auth feature implementation
        self.auth_service: Any | None = None

        # TODO(agent): Filled by agent feature implementation
        self.agent_service: Any | None = None

        # Task layer
        from baize.task.service import TaskEngineService

        self.task_service: TaskEngineService = TaskEngineService(engine=self.db)

    def _init_memory_service(self, config: Settings) -> MemoryServiceInterface | None:
        """Attempt to initialise the configured memory backend.

        Returns None (with a warning) when the memory config is absent or
        incomplete, so the application can still start without a memory service.
        """
        if config.memory.provider != "mem0":
            logger.warning(
                "Unsupported memory provider '%s'; memory service disabled.",
                config.memory.provider,
            )
            return None

        if config.memory.mem0 is None:
            logger.warning(
                "Memory provider is 'mem0' but no 'mem0' sub-config was found; "
                "memory service disabled."
            )
            return None

        try:
            from baize.memory.adapters.mem0 import Mem0Adapter

            return Mem0Adapter(config=config.memory, llm_provider=self.provider_factory)
        except Exception:
            logger.warning(
                "Failed to initialise Mem0Adapter; memory service disabled.",
                exc_info=True,
            )
            return None

    async def close(self) -> None:
        """Release resources held by this container."""
        await self.redis.aclose()
        await self.db.dispose()
