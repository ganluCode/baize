"""Unit tests for AgentConfigService — covers business rules and constraint enforcement."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baize.agent.models import AgentConfig
from baize.agent.repository import AgentConfigRepository
from baize.agent.schemas import AgentCreate, AgentUpdate
from baize.agent.service import AgentConfigService, AgentServiceError
from baize.agent.tools import ToolRegistry
from baize.session.service import SessionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    user_id: uuid.UUID | None = None,
    is_default: bool = False,
    agent_id: uuid.UUID | None = None,
) -> AgentConfig:
    a = AgentConfig()
    a.id = agent_id or uuid.uuid4()
    a.user_id = user_id or uuid.uuid4()
    a.name = "Test Agent"
    a.description = None
    a.system_prompt = "You are helpful."
    a.tools = []
    a.model_config = None
    a.auto_memory_recall = None
    a.shared_memory = None
    a.is_default = is_default
    a.created_at = datetime.now(timezone.utc)
    a.updated_at = datetime.now(timezone.utc)
    return a


def _make_service(
    repo: AgentConfigRepository | None = None,
    session_svc: SessionService | None = None,
) -> AgentConfigService:
    return AgentConfigService(
        repository=repo or AsyncMock(spec=AgentConfigRepository),
        session_service=session_svc or AsyncMock(spec=SessionService),
    )


# ---------------------------------------------------------------------------
# Tool validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_raises_400_on_unknown_tool() -> None:
    """create() raises AgentServiceError(400) when a tool is not registered."""
    repo = AsyncMock(spec=AgentConfigRepository)
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=["save_memory"]):
        data = AgentCreate(name="Agent", system_prompt="hi", tools=["nonexistent_tool"])
        with pytest.raises(AgentServiceError) as exc_info:
            await svc.create(uuid.uuid4(), data)

    assert exc_info.value.status_code == 400
    assert "nonexistent_tool" in str(exc_info.value)
    repo.create.assert_not_called()


@pytest.mark.asyncio
async def test_update_raises_400_on_unknown_tool() -> None:
    """update() raises AgentServiceError(400) when a new tool is not registered."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = agent
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=["save_memory"]):
        data = AgentUpdate(tools=["bad_tool"])
        with pytest.raises(AgentServiceError) as exc_info:
            await svc.update(agent.id, user_id, data)

    assert exc_info.value.status_code == 400
    assert "bad_tool" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_succeeds_with_registered_tools() -> None:
    """create() proceeds when all tools are registered."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.create.return_value = agent
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=["save_memory"]):
        data = AgentCreate(name="Agent", system_prompt="hi", tools=["save_memory"])
        result = await svc.create(user_id, data)

    assert result is agent
    repo.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_succeeds_with_empty_tools() -> None:
    """create() allows an empty tools list (no tools required)."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.create.return_value = agent
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=[]):
        data = AgentCreate(name="Agent", system_prompt="hi", tools=[])
        result = await svc.create(user_id, data)

    assert result is agent


# ---------------------------------------------------------------------------
# Default agent uniqueness on create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_with_is_default_calls_set_default() -> None:
    """create() with is_default=True calls set_default on the repository."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id, is_default=False)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.create.return_value = agent
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=[]):
        data = AgentCreate(name="Agent", system_prompt="hi", tools=[], is_default=True)
        result = await svc.create(user_id, data)

    repo.set_default.assert_called_once_with(user_id, agent.id)
    assert result.is_default is True


@pytest.mark.asyncio
async def test_create_without_is_default_does_not_call_set_default() -> None:
    """create() with is_default=False does not touch set_default."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id, is_default=False)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.create.return_value = agent
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=[]):
        data = AgentCreate(name="Agent", system_prompt="hi", tools=[])
        await svc.create(user_id, data)

    repo.set_default.assert_not_called()


# ---------------------------------------------------------------------------
# Default agent uniqueness on update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_with_is_default_calls_set_default() -> None:
    """update() with is_default=True promotes this agent to default."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id, is_default=False)
    updated_agent = _make_agent(user_id=user_id, agent_id=agent.id, is_default=False)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = agent
    repo.update.return_value = updated_agent
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=[]):
        result = await svc.update(agent.id, user_id, AgentUpdate(is_default=True))

    repo.set_default.assert_called_once_with(user_id, agent.id)
    assert result.is_default is True


@pytest.mark.asyncio
async def test_update_without_is_default_does_not_call_set_default() -> None:
    """update() without is_default change does not call set_default."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id)
    updated = _make_agent(user_id=user_id, agent_id=agent.id)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = agent
    repo.update.return_value = updated
    svc = _make_service(repo=repo)

    with patch.object(ToolRegistry, "get_all_names", return_value=[]):
        await svc.update(agent.id, user_id, AgentUpdate(name="New Name"))

    repo.set_default.assert_not_called()


# ---------------------------------------------------------------------------
# Delete — default agent protection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_raises_400_when_agent_is_default() -> None:
    """delete() raises AgentServiceError(400) when the agent is the default."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id, is_default=True)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = agent
    svc = _make_service(repo=repo)

    with pytest.raises(AgentServiceError) as exc_info:
        await svc.delete(agent.id, user_id)

    assert exc_info.value.status_code == 400
    repo.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_cascades_sessions_then_removes_agent() -> None:
    """delete() calls delete_sessions_by_agent before removing the agent."""
    user_id = uuid.uuid4()
    agent = _make_agent(user_id=user_id, is_default=False)

    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = agent

    session_svc = AsyncMock(spec=SessionService)
    svc = _make_service(repo=repo, session_svc=session_svc)

    await svc.delete(agent.id, user_id)

    session_svc.delete_sessions_by_agent.assert_called_once_with(agent.id)
    repo.delete.assert_called_once_with(agent.id)


# ---------------------------------------------------------------------------
# Ownership enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_raises_404_when_agent_not_found() -> None:
    """get() raises AgentServiceError(404) when no agent exists with that id."""
    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = None
    svc = _make_service(repo=repo)

    with pytest.raises(AgentServiceError) as exc_info:
        await svc.get(uuid.uuid4(), uuid.uuid4())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_raises_404_when_agent_belongs_to_different_user() -> None:
    """get() raises AgentServiceError(404) when agent belongs to another user."""
    agent = _make_agent(user_id=uuid.uuid4())
    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = agent
    svc = _make_service(repo=repo)

    with pytest.raises(AgentServiceError) as exc_info:
        await svc.get(agent.id, uuid.uuid4())  # different user_id

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_raises_404_when_agent_not_found() -> None:
    """delete() raises AgentServiceError(404) when agent does not exist."""
    repo = AsyncMock(spec=AgentConfigRepository)
    repo.get_by_id.return_value = None
    svc = _make_service(repo=repo)

    with pytest.raises(AgentServiceError):
        await svc.delete(uuid.uuid4(), uuid.uuid4())

    repo.delete.assert_not_called()
