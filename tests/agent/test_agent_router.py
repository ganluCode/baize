"""Tests for AgentConfig CRUD API endpoints (F-005)."""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.main import app  # noqa: E402
from baize.agent.schemas import AgentResponse  # noqa: E402
from baize.agent.service import AgentServiceError  # noqa: E402
from baize.user.deps import get_current_user  # noqa: E402
from baize.user.models import UserModel  # noqa: E402

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_USER_ID = uuid.uuid4()
_OTHER_USER_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()


def _make_user(user_id: uuid.UUID | None = None) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id or _USER_ID
    return user


def _make_agent_response(**kwargs) -> AgentResponse:
    defaults = dict(
        id=_AGENT_ID,
        user_id=_USER_ID,
        name="Test Agent",
        description=None,
        system_prompt="You are helpful.",
        tools=["save_memory"],
        model_config=None,
        auto_memory_recall=None,
        shared_memory=None,
        is_default=False,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return AgentResponse(**defaults)


@pytest.fixture
def mock_svc():
    return AsyncMock()


@pytest.fixture
def regular_user():
    return _make_user()


@pytest.fixture
async def client_auth(app, mock_svc, regular_user):
    """Client authenticated as a regular user with agent service mocked."""
    from baize.core.deps import get_agent_config_service

    app.dependency_overrides[get_agent_config_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: regular_user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_config_service, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def client_no_auth(app, mock_svc):
    """Client with no authentication."""
    from baize.core.deps import get_agent_config_service

    app.dependency_overrides[get_agent_config_service] = lambda: mock_svc

    def _raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_config_service, None)
        app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# POST /api/v1/agents
# ---------------------------------------------------------------------------


async def test_create_agent_returns_201(client_auth, mock_svc):
    mock_svc.create.return_value = _make_agent_response(is_default=False)

    response = await client_auth.post(
        "/api/v1/agents",
        json={"name": "Test Agent", "system_prompt": "You are helpful.", "tools": ["save_memory"]},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Test Agent"
    assert body["id"] == str(_AGENT_ID)
    assert body["user_id"] == str(_USER_ID)


async def test_create_agent_calls_service_with_user_id(client_auth, mock_svc, regular_user):
    mock_svc.create.return_value = _make_agent_response()

    await client_auth.post(
        "/api/v1/agents",
        json={"name": "A", "system_prompt": "p", "tools": []},
    )

    call_kwargs = mock_svc.create.call_args
    assert call_kwargs.kwargs["user_id"] == regular_user.id


async def test_create_agent_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.post(
        "/api/v1/agents",
        json={"name": "A", "system_prompt": "p", "tools": []},
    )
    assert response.status_code == 401


async def test_create_agent_invalid_tools_returns_400(client_auth, mock_svc):
    mock_svc.create.side_effect = AgentServiceError("Unknown tools: bad_tool.", status_code=400)

    response = await client_auth.post(
        "/api/v1/agents",
        json={"name": "A", "system_prompt": "p", "tools": ["bad_tool"]},
    )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/v1/agents
# ---------------------------------------------------------------------------


async def test_list_agents_returns_200_with_list(client_auth, mock_svc):
    mock_svc.list.return_value = [_make_agent_response()]

    response = await client_auth.get("/api/v1/agents")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["id"] == str(_AGENT_ID)


async def test_list_agents_empty_returns_empty_list(client_auth, mock_svc):
    mock_svc.list.return_value = []

    response = await client_auth.get("/api/v1/agents")

    assert response.status_code == 200
    assert response.json() == []


async def test_list_agents_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.get("/api/v1/agents")
    assert response.status_code == 401


async def test_list_agents_calls_service_with_user_id(client_auth, mock_svc, regular_user):
    mock_svc.list.return_value = []

    await client_auth.get("/api/v1/agents")

    mock_svc.list.assert_awaited_once_with(user_id=regular_user.id)


# ---------------------------------------------------------------------------
# GET /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


async def test_get_agent_returns_200_when_found(client_auth, mock_svc):
    mock_svc.get.return_value = _make_agent_response()

    response = await client_auth.get(f"/api/v1/agents/{_AGENT_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(_AGENT_ID)


async def test_get_agent_not_found_returns_404(client_auth, mock_svc):
    mock_svc.get.side_effect = AgentServiceError("Agent not found.", status_code=404)

    response = await client_auth.get(f"/api/v1/agents/{_AGENT_ID}")

    assert response.status_code == 404


async def test_get_agent_other_user_returns_404(client_auth, mock_svc):
    """Other user's agent returns 404, not 403, to avoid leaking existence."""
    mock_svc.get.side_effect = AgentServiceError("Agent not found.", status_code=404)

    response = await client_auth.get(f"/api/v1/agents/{_AGENT_ID}")

    assert response.status_code == 404


async def test_get_agent_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.get(f"/api/v1/agents/{_AGENT_ID}")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# PUT /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


async def test_update_agent_returns_200(client_auth, mock_svc):
    mock_svc.update.return_value = _make_agent_response(name="Updated Name")

    response = await client_auth.put(
        f"/api/v1/agents/{_AGENT_ID}",
        json={"name": "Updated Name"},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Updated Name"


async def test_update_agent_not_found_returns_404(client_auth, mock_svc):
    mock_svc.update.side_effect = AgentServiceError("Agent not found.", status_code=404)

    response = await client_auth.put(
        f"/api/v1/agents/{_AGENT_ID}",
        json={"name": "X"},
    )

    assert response.status_code == 404


async def test_update_agent_invalid_tools_returns_400(client_auth, mock_svc):
    mock_svc.update.side_effect = AgentServiceError("Unknown tools: bad_tool.", status_code=400)

    response = await client_auth.put(
        f"/api/v1/agents/{_AGENT_ID}",
        json={"tools": ["bad_tool"]},
    )

    assert response.status_code == 400


async def test_update_agent_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.put(
        f"/api/v1/agents/{_AGENT_ID}",
        json={"name": "X"},
    )
    assert response.status_code == 401


async def test_update_agent_calls_service_with_user_id(client_auth, mock_svc, regular_user):
    mock_svc.update.return_value = _make_agent_response()

    await client_auth.put(f"/api/v1/agents/{_AGENT_ID}", json={"name": "X"})

    call_kwargs = mock_svc.update.call_args
    assert call_kwargs.kwargs["user_id"] == regular_user.id
    assert call_kwargs.kwargs["agent_id"] == _AGENT_ID


# ---------------------------------------------------------------------------
# DELETE /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


async def test_delete_agent_returns_204(client_auth, mock_svc):
    mock_svc.delete.return_value = None

    response = await client_auth.delete(f"/api/v1/agents/{_AGENT_ID}")

    assert response.status_code == 204


async def test_delete_agent_not_found_returns_404(client_auth, mock_svc):
    mock_svc.delete.side_effect = AgentServiceError("Agent not found.", status_code=404)

    response = await client_auth.delete(f"/api/v1/agents/{_AGENT_ID}")

    assert response.status_code == 404


async def test_delete_default_agent_returns_400(client_auth, mock_svc):
    mock_svc.delete.side_effect = AgentServiceError(
        "Cannot delete the default agent.", status_code=400
    )

    response = await client_auth.delete(f"/api/v1/agents/{_AGENT_ID}")

    assert response.status_code == 400


async def test_delete_agent_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.delete(f"/api/v1/agents/{_AGENT_ID}")
    assert response.status_code == 401


async def test_delete_agent_calls_service_with_correct_args(client_auth, mock_svc, regular_user):
    mock_svc.delete.return_value = None

    await client_auth.delete(f"/api/v1/agents/{_AGENT_ID}")

    mock_svc.delete.assert_awaited_once_with(agent_id=_AGENT_ID, user_id=regular_user.id)
