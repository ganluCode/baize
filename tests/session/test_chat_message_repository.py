"""Tests for ChatMessageRepository — covers write and query operations."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.session.models import ChatMessageModel, MessageRole
from baize.session.repository import ChatMessageRepository


def _make_message(
    session_id: uuid.UUID | None = None,
    role: MessageRole = MessageRole.user,
    content: str = "hello",
    created_at: datetime | None = None,
) -> ChatMessageModel:
    msg = ChatMessageModel()
    msg.id = uuid.uuid4()
    msg.session_id = session_id or uuid.uuid4()
    msg.user_id = uuid.uuid4()
    msg.role = role
    msg.content = content
    msg.tool_calls = None
    msg.tool_name = None
    msg.token_usage = None
    msg.created_at = created_at or datetime.now(UTC)
    return msg


@pytest.fixture
def db_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def repo(db_session: AsyncMock) -> ChatMessageRepository:
    return ChatMessageRepository(db_session)


@pytest.mark.asyncio
async def test_create_adds_and_returns_message(repo: ChatMessageRepository, db_session: AsyncMock) -> None:
    """create must add the message to the session and return the refreshed object."""
    session_id = uuid.uuid4()
    user_id = uuid.uuid4()

    created_msg = _make_message(session_id=session_id, role=MessageRole.user, content="hi")

    async def mock_refresh(obj):
        obj.id = created_msg.id
        obj.created_at = created_msg.created_at

    db_session.refresh.side_effect = mock_refresh

    result = await repo.create(
        session_id=session_id,
        user_id=user_id,
        role=MessageRole.user,
        content="hi",
    )

    db_session.add.assert_called_once()
    db_session.commit.assert_called_once()
    db_session.refresh.assert_called_once()
    assert result.session_id == session_id
    assert result.role == MessageRole.user
    assert result.content == "hi"
    assert result.tool_calls is None
    assert result.tool_name is None
    assert result.token_usage is None


@pytest.mark.asyncio
async def test_create_with_optional_fields(repo: ChatMessageRepository, db_session: AsyncMock) -> None:
    """create passes optional fields through to the model."""
    session_id = uuid.uuid4()
    tool_calls = {"name": "search", "args": {}}
    token_usage = {"input": 10, "output": 20}

    db_session.refresh.side_effect = AsyncMock()

    result = await repo.create(
        session_id=session_id,
        user_id=uuid.uuid4(),
        role=MessageRole.tool,
        content="result",
        tool_calls=tool_calls,
        tool_name="search",
        token_usage=token_usage,
    )

    assert result.tool_calls == tool_calls
    assert result.tool_name == "search"
    assert result.token_usage == token_usage


@pytest.mark.asyncio
async def test_list_by_session_returns_tuple(repo: ChatMessageRepository, db_session: AsyncMock) -> None:
    """list_by_session returns a (list, int) tuple."""
    session_id = uuid.uuid4()
    msg = _make_message(session_id=session_id)

    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = [msg]
    count_result = MagicMock()
    count_result.scalar_one.return_value = 1

    db_session.execute.side_effect = [count_result, items_result]

    result = await repo.list_by_session(session_id=session_id)

    assert isinstance(result, tuple)
    items, total = result
    assert isinstance(items, list)
    assert isinstance(total, int)
    assert total == 1
    assert items == [msg]


@pytest.mark.asyncio
async def test_list_by_session_empty(repo: ChatMessageRepository, db_session: AsyncMock) -> None:
    """list_by_session returns ([], 0) when session has no messages."""
    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = []
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0

    db_session.execute.side_effect = [count_result, items_result]

    items, total = await repo.list_by_session(session_id=uuid.uuid4())

    assert items == []
    assert total == 0


@pytest.mark.asyncio
async def test_get_recent_returns_list(repo: ChatMessageRepository, db_session: AsyncMock) -> None:
    """get_recent returns a list of ChatMessageModel."""
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    msg1 = _make_message(session_id=session_id, created_at=now - timedelta(seconds=10))
    msg2 = _make_message(session_id=session_id, created_at=now)

    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = [msg1, msg2]

    db_session.execute.return_value = items_result

    result = await repo.get_recent(session_id=session_id, limit=10)

    assert isinstance(result, list)
    assert result == [msg1, msg2]


@pytest.mark.asyncio
async def test_get_recent_ascending_order(repo: ChatMessageRepository, db_session: AsyncMock) -> None:
    """get_recent returns messages in ascending created_at order."""
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    older = _make_message(session_id=session_id, created_at=now - timedelta(minutes=5))
    newer = _make_message(session_id=session_id, created_at=now)

    items_result = MagicMock()
    # Simulate DB returning already-sorted ascending result
    items_result.scalars.return_value.all.return_value = [older, newer]

    db_session.execute.return_value = items_result

    result = await repo.get_recent(session_id=session_id, limit=2)

    assert result[0].created_at <= result[1].created_at


@pytest.mark.asyncio
async def test_get_recent_empty(repo: ChatMessageRepository, db_session: AsyncMock) -> None:
    """get_recent returns an empty list when session has no messages."""
    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = []

    db_session.execute.return_value = items_result

    result = await repo.get_recent(session_id=uuid.uuid4(), limit=5)

    assert result == []
