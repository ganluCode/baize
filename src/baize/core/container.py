"""Application-wide service container for Baize (manual dependency injection)."""

from typing import Any

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine

from baize.core.config import Settings
from baize.user.jwt_service import JWTService


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

        # TODO(memory): Filled by memory feature implementation
        self.memory_service: Any | None = None

        # TODO(session): Filled by session feature implementation
        self.session_service: Any | None = None

        # TODO(user): Filled by user feature implementation
        self.user_service: Any | None = None

        # TODO(auth): Filled by auth feature implementation
        self.auth_service: Any | None = None

        # TODO(agent): Filled by agent feature implementation
        self.agent_service: Any | None = None

        # TODO(task): Filled by task feature implementation
        self.task_service: Any | None = None

    async def close(self) -> None:
        """Release resources held by this container."""
        await self.redis.aclose()
        await self.db.dispose()
