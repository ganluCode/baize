"""Integration tests for default agent creation (F-017).

Verifies that a newly created user automatically has a default agent
with is_default=True and a non-empty system_prompt.

Tests:
- GET /api/v1/agents after user creation returns a list containing one
  is_default=True agent whose system_prompt is non-empty.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.agent.schemas import AgentResponse  # noqa: E402
from baize.core.deps import get_agent_config_service  # noqa: E402
from baize.main import app  # noqa: E402
from baize.user.deps import get_current_user  # noqa: E402
from baize.user.models import UserModel  # noqa: E402

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NEW_USER_ID = uuid.uuid4()
_DEFAULT_AGENT_ID = uuid.uuid4()


def _make_user(user_id: uuid.UUID) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id
    user.name = "ganlu"
    return user


def _make_default_agent_response(system_prompt: str = "You are 白泽（Baize）...") -> AgentResponse:
    return AgentResponse(
        id=_DEFAULT_AGENT_ID,
        user_id=_NEW_USER_ID,
        name="白泽（Baize）",
        description="你的个人 AI 助理",
        system_prompt=system_prompt,
        tools=["save_memory", "search_memory", "create_task", "list_tasks", "complete_task"],
        model_config=None,
        auto_memory_recall=None,
        shared_memory=None,
        is_default=True,
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.fixture
def new_user():
    return _make_user(_NEW_USER_ID)


@pytest.fixture
def mock_agent_svc():
    return AsyncMock()


@pytest.fixture
async def client_new_user(app, mock_agent_svc, new_user):
    """Client authenticated as a freshly-created user."""
    app.dependency_overrides[get_agent_config_service] = lambda: mock_agent_svc
    app.dependency_overrides[get_current_user] = lambda: new_user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_config_service, None)
        app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# Default agent existence after user creation
# ---------------------------------------------------------------------------


async def test_new_user_has_default_agent_in_list(client_new_user, mock_agent_svc):
    """GET /api/v1/agents for a new user returns exactly one is_default=True agent."""
    default_agent = _make_default_agent_response()
    mock_agent_svc.list.return_value = [default_agent]

    response = await client_new_user.get("/api/v1/agents")

    assert response.status_code == 200
    agents = response.json()
    assert len(agents) >= 1
    default_agents = [a for a in agents if a["is_default"] is True]
    assert len(default_agents) == 1


async def test_new_user_default_agent_has_non_empty_system_prompt(client_new_user, mock_agent_svc):
    """The default agent's system_prompt must not be empty."""
    default_agent = _make_default_agent_response(system_prompt="You are 白泽（Baize）, your personal AI assistant.")
    mock_agent_svc.list.return_value = [default_agent]

    response = await client_new_user.get("/api/v1/agents")

    assert response.status_code == 200
    agents = response.json()
    default_agents = [a for a in agents if a["is_default"] is True]
    assert len(default_agents) == 1
    assert default_agents[0]["system_prompt"]  # non-empty string is truthy


async def test_new_user_default_agent_is_default_true(client_new_user, mock_agent_svc):
    """The agent returned for a new user has is_default=True."""
    default_agent = _make_default_agent_response()
    mock_agent_svc.list.return_value = [default_agent]

    response = await client_new_user.get("/api/v1/agents")

    assert response.status_code == 200
    agents = response.json()
    assert any(a["is_default"] is True for a in agents)


# ---------------------------------------------------------------------------
# AgentConfigService.create_default_agent integration
# ---------------------------------------------------------------------------


async def test_create_default_agent_service_called_on_user_creation():
    """Verifies that AgentConfigService.create_default_agent is invoked during user creation.

    This tests the service layer integration: UserService.create_user calls
    create_default_agent when an AgentConfigService is provided.
    """
    from baize.agent.service import AgentConfigService
    from baize.user.repository import UserRepository
    from baize.user.schemas import UserCreateRequest
    from baize.user.service import UserService

    mock_user_repo = AsyncMock(spec=UserRepository)
    mock_user = MagicMock()
    mock_user.id = _NEW_USER_ID
    mock_user.email = "test@example.com"
    mock_user.name = "testuser"
    mock_user.role = "user"
    mock_user.avatar = None
    mock_user.preferences = None
    mock_user.is_active = True
    mock_user.created_at = _NOW
    mock_user.updated_at = _NOW
    mock_user_repo.create.return_value = mock_user

    mock_agent_svc = AsyncMock(spec=AgentConfigService)
    mock_agent_svc.create_default_agent.return_value = _make_default_agent_response()

    user_svc = UserService(user_repo=mock_user_repo, agent_config_service=mock_agent_svc)

    data = UserCreateRequest(email="test@example.com", name="testuser", password="secret123")
    result = await user_svc.create_user(data)

    # create_default_agent must have been called with the new user's id
    mock_agent_svc.create_default_agent.assert_awaited_once_with(_NEW_USER_ID)
    assert result.email == "test@example.com"
