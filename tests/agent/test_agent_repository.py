"""Tests for AgentConfigRepository — covers business logic and query behavior."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.agent.models import AgentConfig
from baize.agent.repository import AgentConfigRepository
from baize.agent.schemas import AgentUpdate


def _make_agent(
    user_id: uuid.UUID | None = None,
    is_default: bool = False,
) -> AgentConfig:
    a = AgentConfig()
    a.id = uuid.uuid4()
    a.user_id = user_id or uuid.uuid4()
    a.name = "Test Agent"
    a.description = None
    a.agent_type = "chat"
    a.is_enabled = True
    a.prompts = {"behavior": "You are a helpful assistant."}
    a.tools = {"builtin": ["save_memory"], "mcp_servers": [], "skills": []}
    a.model_config_json = None
    a.sub_agents = None
    a.memory_config = {"auto_recall": True, "shared": True, "top_k": 5}
    a.guardrails = {"max_tool_calls": 10, "timeout_seconds": 120}
    a.created_at = datetime.now(UTC)
    a.updated_at = datetime.now(UTC)
    return a


@pytest.fixture
def db_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def repo(db_session: AsyncMock) -> AgentConfigRepository:
    return AgentConfigRepository(db_session)


# --- list_by_user ---


@pytest.mark.asyncio
async def test_list_by_user_returns_only_matching_user(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """list_by_user must only return agents belonging to the specified user."""
    user_id = uuid.uuid4()
    matching = _make_agent(user_id=user_id)
    other = _make_agent(user_id=uuid.uuid4())

    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [matching]
    db_session.execute.return_value = result_mock

    items = await repo.list_by_user(user_id)

    assert items == [matching]
    assert other not in items


@pytest.mark.asyncio
async def test_list_by_user_returns_empty_list(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """list_by_user returns an empty list when no agents exist."""
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    db_session.execute.return_value = result_mock

    items = await repo.list_by_user(uuid.uuid4())

    assert items == []


# --- get_default_by_user ---


@pytest.mark.asyncio
async def test_get_default_by_user_returns_default_agent(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """get_default_by_user returns the agent pointed to by default_agent_id."""
    user_id = uuid.uuid4()
    default_agent = _make_agent(user_id=user_id)
    default_agent_id = default_agent.id

    # First execute: get default_agent_id from system_users
    user_result_mock = MagicMock()
    user_result_mock.scalar_one_or_none.return_value = default_agent_id

    # Second execute: get_by_id for the agent
    agent_result_mock = MagicMock()
    agent_result_mock.scalar_one_or_none.return_value = default_agent

    db_session.execute.side_effect = [user_result_mock, agent_result_mock]

    result = await repo.get_default_by_user(user_id)

    assert result is default_agent


@pytest.mark.asyncio
async def test_get_default_by_user_returns_none_when_no_default(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """get_default_by_user returns None when no default agent exists."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    db_session.execute.return_value = result_mock

    result = await repo.get_default_by_user(uuid.uuid4())

    assert result is None


# --- set_default ---


@pytest.mark.asyncio
async def test_set_default_clears_all_then_sets_target(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """set_default verifies ownership then updates system_users.default_agent_id."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    # First execute: verify agent belongs to user (SELECT AgentConfig.id)
    check_result = MagicMock()
    check_result.scalar_one_or_none.return_value = agent_id

    # Second execute: UPDATE system_users
    update_result = MagicMock()
    db_session.execute.side_effect = [check_result, update_result]

    await repo.set_default(user_id, agent_id)

    # Two executes: select to check ownership + update system_users
    assert db_session.execute.call_count == 2
    # commit called
    db_session.commit.assert_called()


@pytest.mark.asyncio
async def test_set_default_does_nothing_when_agent_not_found(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """set_default does not update when agent_id is not found for user."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    # First (and only) execute: SELECT to verify ownership returns None
    check_result = MagicMock()
    check_result.scalar_one_or_none.return_value = None
    db_session.execute.return_value = check_result

    await repo.set_default(user_id, agent_id)

    # Only one execute: ownership check; no UPDATE executed, no commit
    assert db_session.execute.call_count == 1
    db_session.commit.assert_not_called()


# --- update ---


@pytest.mark.asyncio
async def test_update_returns_none_when_agent_not_found(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """update returns None when the specified agent does not exist."""
    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = None
    db_session.execute.return_value = get_result

    result = await repo.update(uuid.uuid4(), AgentUpdate(name="New Name"))

    assert result is None
    db_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_update_applies_provided_fields(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """update applies only non-None fields from AgentUpdate."""
    existing = _make_agent()

    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = existing
    db_session.execute.return_value = get_result

    updated = await repo.update(existing.id, AgentUpdate(name="Updated Name"))

    assert updated is existing
    assert existing.name == "Updated Name"
    db_session.commit.assert_called_once()


# --- delete ---


@pytest.mark.asyncio
async def test_delete_returns_true_when_agent_exists(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """delete returns True when the agent exists and is deleted."""
    existing = _make_agent()

    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = existing
    db_session.execute.return_value = get_result

    result = await repo.delete(existing.id)

    assert result is True
    db_session.delete.assert_called_once_with(existing)
    db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_returns_false_when_agent_not_found(
    repo: AgentConfigRepository, db_session: AsyncMock
) -> None:
    """delete returns False when no agent with that id exists."""
    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = None
    db_session.execute.return_value = get_result

    result = await repo.delete(uuid.uuid4())

    assert result is False
    db_session.delete.assert_not_called()
    db_session.commit.assert_not_called()
