"""Tests for LLM router endpoints (src/baize/llm/router.py)."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.core.deps import get_provider_factory  # noqa: E402
from baize.llm.provider import (  # noqa: E402
    ProviderFactory,
    ProviderNotFoundError,
    ProviderUnavailableError,
)
from baize.user.deps import get_current_user  # noqa: E402
from baize.user.models import UserModel  # noqa: E402


def _make_user() -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = "00000000-0000-0000-0000-000000000001"
    user.role = "user"
    return user


def _make_provider_factory(
    *,
    providers: list[dict] | None = None,
    unavailable: set[str] | None = None,
) -> MagicMock:
    """Build a mock ProviderFactory with configurable provider list."""
    factory = MagicMock(spec=ProviderFactory)

    if providers is None:
        providers = [
            {
                "name": "doubao",
                "type": "openai",
                "status": "available",
                "models": [{"id": "doubao-pro-256k", "usage": ["chat"]}],
            },
            {
                "name": "claude",
                "type": "anthropic",
                "status": "unavailable",
                "models": [{"id": "claude-sonnet-4-6", "usage": ["chat", "reasoning"]}],
            },
        ]

    factory.list_providers.return_value = providers
    return factory


@pytest.fixture
def mock_user() -> UserModel:
    return _make_user()


@pytest.fixture
def mock_factory() -> MagicMock:
    return _make_provider_factory()


@pytest.fixture
async def client(app, mock_user, mock_factory):
    """Async client with auth and provider_factory dependencies overridden."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_provider_factory] = lambda: mock_factory
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_provider_factory, None)


@pytest.fixture
async def client_no_auth(app, mock_factory):
    """Async client where auth raises 401 (mirrors project convention)."""

    def _raise_401() -> None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_provider_factory] = lambda: mock_factory
    app.dependency_overrides[get_current_user] = _raise_401
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_provider_factory, None)
    app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# GET /api/v1/llm/providers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_providers_returns_200(client):
    response = await client.get("/api/v1/llm/providers")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_providers_response_schema(client):
    response = await client.get("/api/v1/llm/providers")
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    first = data[0]
    assert "name" in first
    assert "type" in first
    assert "status" in first
    assert "models" in first


@pytest.mark.asyncio
async def test_list_providers_no_sensitive_fields(client):
    response = await client.get("/api/v1/llm/providers")
    data = response.json()
    for provider in data:
        assert "api_key" not in provider
        assert "base_url" not in provider


@pytest.mark.asyncio
async def test_list_providers_status_values(client):
    response = await client.get("/api/v1/llm/providers")
    data = response.json()
    statuses = {p["status"] for p in data}
    assert statuses <= {"available", "unavailable"}


@pytest.mark.asyncio
async def test_list_providers_requires_auth(client_no_auth):
    """With auth raising 401, GET /api/v1/llm/providers should return 401."""
    response = await client_no_auth.get("/api/v1/llm/providers")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/llm/test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_test_unknown_provider(client, mock_factory):
    mock_factory.get_chat_model.side_effect = ProviderNotFoundError("nonexistent")

    response = await client.post(
        "/api/v1/llm/test", json={"provider": "nonexistent", "model": "some-model"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "error" in body
    assert body["error"]


@pytest.mark.asyncio
async def test_llm_test_unavailable_provider(client, mock_factory):
    mock_factory.get_chat_model.side_effect = ProviderUnavailableError("claude")

    response = await client.post(
        "/api/v1/llm/test", json={"provider": "claude", "model": "claude-sonnet-4-6"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"]


@pytest.mark.asyncio
async def test_llm_test_success(client, mock_factory):
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = MagicMock(content="pong")
    mock_factory.get_chat_model.return_value = mock_model

    response = await client.post(
        "/api/v1/llm/test",
        json={"provider": "doubao", "model": "doubao-pro-256k"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert isinstance(body["latency_ms"], int)
    assert body["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_llm_test_requires_auth(client_no_auth):
    """With auth raising 401, POST /api/v1/llm/test should return 401."""
    response = await client_no_auth.post(
        "/api/v1/llm/test",
        json={"provider": "doubao", "model": "doubao-pro-256k"},
    )
    assert response.status_code == 401
