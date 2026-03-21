"""Shared test fixtures for Baize test suite."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Set required env vars before any baize import that may trigger Settings()
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.main import app as _app  # noqa: E402


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock AsyncEngine — avoids real database connections in tests."""
    mock = MagicMock()
    mock.dispose = AsyncMock()
    return mock


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis async client — avoids real Redis connections in tests."""
    mock = AsyncMock()
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def app(mock_db: MagicMock, mock_redis: AsyncMock):
    """FastAPI app with infrastructure dependencies mocked out."""
    with (
        patch("baize.core.container.aioredis.from_url", return_value=mock_redis),
        patch("baize.core.database.engine", mock_db),
    ):
        yield _app


@pytest.fixture
async def async_client(app):
    """Async HTTP client wired to the test app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
