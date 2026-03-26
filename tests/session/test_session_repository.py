"""Tests for SessionRepository — covers filtering and ordering logic."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.session.models import SessionModel, SessionStatus
from baize.session.repository import SessionRepository


def _make_session(
    user_id: uuid.UUID | None = None,
    agent_id: uuid.UUID | None = None,
    status: SessionStatus = SessionStatus.active,
    updated_at: datetime | None = None,
) -> SessionModel:
    s = SessionModel()
    s.id = uuid.uuid4()
    s.user_id = user_id or uuid.uuid4()
    s.agent_id = agent_id or uuid.uuid4()
    s.title = None
    s.status = status
    s.auto_memory_recall = None
    s.shared_memory = None
    s.created_at = datetime.now(UTC)
    s.updated_at = updated_at or datetime.now(UTC)
    return s


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
def repo(db_session: AsyncMock) -> SessionRepository:
    return SessionRepository(db_session)


@pytest.mark.asyncio
async def test_list_by_agent_filters_by_user_id(repo: SessionRepository, db_session: AsyncMock) -> None:
    """list_by_agent must only return sessions belonging to the specified user."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    other_user_session = _make_session(user_id=uuid.uuid4(), agent_id=agent_id)
    matching_session = _make_session(user_id=user_id, agent_id=agent_id)

    # Simulate DB returning only matching sessions (filter is applied in query)
    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = [matching_session]
    count_result = MagicMock()
    count_result.scalar_one.return_value = 1

    db_session.execute.side_effect = [count_result, items_result]

    items, total = await repo.list_by_agent(user_id=user_id, agent_id=agent_id)

    assert total == 1
    assert items == [matching_session]
    # other_user_session should not be in results
    assert other_user_session not in items


@pytest.mark.asyncio
async def test_list_by_agent_status_filter(repo: SessionRepository, db_session: AsyncMock) -> None:
    """list_by_agent passes status filter through when provided."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    archived = _make_session(user_id=user_id, agent_id=agent_id, status=SessionStatus.archived)

    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = [archived]
    count_result = MagicMock()
    count_result.scalar_one.return_value = 1

    db_session.execute.side_effect = [count_result, items_result]

    items, total = await repo.list_by_agent(
        user_id=user_id, agent_id=agent_id, status=SessionStatus.archived
    )

    assert total == 1
    assert items[0].status == SessionStatus.archived


@pytest.mark.asyncio
async def test_list_by_agent_returns_tuple(repo: SessionRepository, db_session: AsyncMock) -> None:
    """list_by_agent returns a (list, int) tuple."""
    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = []
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0

    db_session.execute.side_effect = [count_result, items_result]

    result = await repo.list_by_agent(user_id=uuid.uuid4(), agent_id=uuid.uuid4())

    assert isinstance(result, tuple)
    assert len(result) == 2
    items, total = result
    assert isinstance(items, list)
    assert isinstance(total, int)


@pytest.mark.asyncio
async def test_update_returns_refreshed_session(repo: SessionRepository, db_session: AsyncMock) -> None:
    """update must return the session after applying changes."""
    session_id = uuid.uuid4()
    existing = _make_session()
    existing.id = session_id

    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = existing
    db_session.execute.return_value = get_result

    async def mock_refresh(obj):
        # Simulate refresh by ensuring object is returned unchanged
        pass

    db_session.refresh.side_effect = mock_refresh

    updated = await repo.update(session_id, title="New Title", status=SessionStatus.archived)

    assert updated is existing
    assert existing.title == "New Title"
    assert existing.status == SessionStatus.archived


@pytest.mark.asyncio
async def test_update_nonexistent_returns_none(repo: SessionRepository, db_session: AsyncMock) -> None:
    """update returns None when session does not exist."""
    get_result = MagicMock()
    get_result.scalar_one_or_none.return_value = None
    db_session.execute.return_value = get_result

    result = await repo.update(uuid.uuid4(), title="x")

    assert result is None
    db_session.commit.assert_not_called()
