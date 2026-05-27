"""API integration tests for agent CRUD endpoints (F-017).

Covers:
- POST /api/v1/agents → 201 with correct body
- GET /api/v1/agents → only current user's agents (user isolation)
- PUT /api/v1/agents/{id} → 200 with updated data
- DELETE /api/v1/agents/{id} → 204
- Accessing another user's agent → 404
"""

import os
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.agent.schemas import AgentResponse  # noqa: E402
from baize.agent.service import AgentServiceError  # noqa: E402
from baize.core.deps import get_agent_config_service  # noqa: E402
from baize.user.deps import get_current_user  # noqa: E402
from baize.user.models import UserModel  # noqa: E402

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_USER_A_ID = uuid.uuid4()
_USER_B_ID = uuid.uuid4()
_AGENT_A_ID = uuid.uuid4()
_AGENT_B_ID = uuid.uuid4()


def _make_user(user_id: uuid.UUID) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id
    return user


def _agent_resp(agent_id: uuid.UUID, user_id: uuid.UUID, **kwargs) -> AgentResponse:
    defaults = dict(
        id=agent_id,
        user_id=user_id,
        name="Test Agent",
        description=None,
        agent_type="chat",
        prompts={"behavior": "You are helpful."},
        tools={"builtin": ["save_memory"], "mcp_servers": [], "skills": []},
        model_config=None,
        sub_agents=None,
        memory_config={"auto_recall": True, "shared": True, "top_k": 5},
        guardrails={"max_tool_calls": 10, "timeout_seconds": 120},
        is_enabled=True,
        created_at=_NOW,
        updated_at=_NOW,
    )
    # remove old fields that no longer exist
    kwargs.pop("system_prompt", None)
    defaults.update(kwargs)
    return AgentResponse(**defaults)


@pytest.fixture
def mock_svc():
    return AsyncMock()


@pytest.fixture
def user_a():
    return _make_user(_USER_A_ID)


@pytest.fixture
def user_b():
    return _make_user(_USER_B_ID)


@pytest.fixture
async def client_a(app, mock_svc, user_a):
    """Authenticated client for user A."""
    app.dependency_overrides[get_agent_config_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: user_a
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_config_service, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def client_b(app, mock_svc, user_b):
    """Authenticated client for user B."""
    app.dependency_overrides[get_agent_config_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: user_b
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_config_service, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def client_invalid_key(app, mock_svc):
    """Client that will reject authentication (simulates invalid API key)."""
    app.dependency_overrides[get_agent_config_service] = lambda: mock_svc

    def _reject():
        raise HTTPException(status_code=401, detail="Invalid API key")

    app.dependency_overrides[get_current_user] = _reject
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_config_service, None)
        app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# POST /api/v1/agents
# ---------------------------------------------------------------------------


async def test_create_agent_returns_201(client_a, mock_svc):
    mock_svc.create.return_value = _agent_resp(_AGENT_A_ID, _USER_A_ID)

    response = await client_a.post(
        "/api/v1/agents",
        json={"name": "Test Agent", "prompts": {"behavior": "You are helpful."}},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == str(_AGENT_A_ID)
    assert body["user_id"] == str(_USER_A_ID)
    assert body["name"] == "Test Agent"


async def test_create_agent_without_valid_key_returns_401(client_invalid_key):
    response = await client_invalid_key.post(
        "/api/v1/agents",
        json={"name": "A", "system_prompt": "p", "tools": []},
    )
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# GET /api/v1/agents — user isolation
# ---------------------------------------------------------------------------


async def test_list_agents_returns_only_current_user_agents(app, mock_svc, user_a, user_b):
    """User A and User B each see only their own agents."""
    agent_a = _agent_resp(_AGENT_A_ID, _USER_A_ID, name="Agent A")
    agent_b = _agent_resp(_AGENT_B_ID, _USER_B_ID, name="Agent B")

    async def _get_for_user():
        # Return different results based on which user is making the request
        pass

    # --- User A sees only agent_a ---
    app.dependency_overrides[get_agent_config_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: user_a
    mock_svc.list.return_value = [agent_a]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/agents")

    assert resp.status_code == 200
    ids_a = [item["id"] for item in resp.json()]
    assert str(_AGENT_A_ID) in ids_a
    assert str(_AGENT_B_ID) not in ids_a

    # --- User B sees only agent_b ---
    app.dependency_overrides[get_current_user] = lambda: user_b
    mock_svc.list.return_value = [agent_b]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/agents")

    assert resp.status_code == 200
    ids_b = [item["id"] for item in resp.json()]
    assert str(_AGENT_B_ID) in ids_b
    assert str(_AGENT_A_ID) not in ids_b

    app.dependency_overrides.pop(get_agent_config_service, None)
    app.dependency_overrides.pop(get_current_user, None)


async def test_list_calls_service_with_requesting_user_id(client_a, mock_svc, user_a):
    mock_svc.list.return_value = []

    await client_a.get("/api/v1/agents")

    mock_svc.list.assert_awaited_once_with(user_id=user_a.id)


# ---------------------------------------------------------------------------
# PUT /api/v1/agents/{agent_id} — returns updated data
# ---------------------------------------------------------------------------


async def test_update_agent_returns_updated_data(client_a, mock_svc):
    updated = _agent_resp(_AGENT_A_ID, _USER_A_ID, name="Updated Name")
    mock_svc.update.return_value = updated

    response = await client_a.put(
        f"/api/v1/agents/{_AGENT_A_ID}",
        json={"name": "Updated Name"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Updated Name"
    assert body["id"] == str(_AGENT_A_ID)


async def test_update_other_user_agent_returns_404(client_a, mock_svc):
    mock_svc.update.side_effect = AgentServiceError("Agent not found.", status_code=404)

    response = await client_a.put(
        f"/api/v1/agents/{_AGENT_B_ID}",
        json={"name": "Hijack"},
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


async def test_delete_agent_returns_204(client_a, mock_svc):
    mock_svc.delete.return_value = None

    response = await client_a.delete(f"/api/v1/agents/{_AGENT_A_ID}")

    assert response.status_code == 204


async def test_delete_other_user_agent_returns_404(client_a, mock_svc):
    mock_svc.delete.side_effect = AgentServiceError("Agent not found.", status_code=404)

    response = await client_a.delete(f"/api/v1/agents/{_AGENT_B_ID}")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/agents/{agent_id} — other user's agent returns 404
# ---------------------------------------------------------------------------


async def test_get_other_user_agent_returns_404(client_a, mock_svc):
    """Accessing another user's agent returns 404, not 403, to avoid leaking existence."""
    mock_svc.get.side_effect = AgentServiceError("Agent not found.", status_code=404)

    response = await client_a.get(f"/api/v1/agents/{_AGENT_B_ID}")

    assert response.status_code == 404
