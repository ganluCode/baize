"""Tests for SessionService — covers save_message, get_history, get_or_create_session."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from baize.session.models import ChatMessageModel, MessageRole, SessionModel, SessionStatus
from baize.session.repository import ChatMessageRepository, SessionRepository
from baize.session.service import SessionService


def _make_session(
    session_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    agent_id: uuid.UUID | None = None,
) -> SessionModel:
    s = SessionModel()
    s.id = session_id or uuid.uuid4()
    s.user_id = user_id or uuid.uuid4()
    s.agent_id = agent_id or uuid.uuid4()
    s.title = None
    s.status = SessionStatus.active
    s.auto_memory_recall = None
    s.shared_memory = None
    s.created_at = datetime.now(timezone.utc)
    s.updated_at = datetime.now(timezone.utc)
    return s


def _make_message(session_id: uuid.UUID | None = None) -> ChatMessageModel:
    msg = ChatMessageModel()
    msg.id = uuid.uuid4()
    msg.session_id = session_id or uuid.uuid4()
    msg.user_id = uuid.uuid4()
    msg.role = MessageRole.user
    msg.content = "hello"
    msg.tool_calls = None
    msg.tool_name = None
    msg.token_usage = None
    msg.created_at = datetime.now(timezone.utc)
    return msg


@pytest.fixture
def session_repo() -> AsyncMock:
    repo = AsyncMock(spec=SessionRepository)
    return repo


@pytest.fixture
def message_repo() -> AsyncMock:
    repo = AsyncMock(spec=ChatMessageRepository)
    return repo


@pytest.fixture
def service(session_repo: AsyncMock, message_repo: AsyncMock) -> SessionService:
    return SessionService(session_repo=session_repo, message_repo=message_repo)


# --- save_message ---


@pytest.mark.asyncio
async def test_save_message_creates_message_and_updates_session(
    service: SessionService,
    session_repo: AsyncMock,
    message_repo: AsyncMock,
) -> None:
    """save_message calls create on message repo and update on session repo."""
    session_id = uuid.uuid4()
    user_id = uuid.uuid4()
    expected_msg = _make_message(session_id=session_id)
    message_repo.create.return_value = expected_msg
    session_repo.update.return_value = _make_session(session_id=session_id)

    result = await service.save_message(
        session_id=session_id,
        user_id=user_id,
        role=MessageRole.user,
        content="hello",
        tool_calls=None,
        tool_name=None,
        token_usage=None,
    )

    message_repo.create.assert_called_once_with(
        session_id=session_id,
        user_id=user_id,
        role=MessageRole.user,
        content="hello",
        tool_calls=None,
        tool_name=None,
        token_usage=None,
    )
    session_repo.update.assert_called_once()
    assert result is expected_msg


@pytest.mark.asyncio
async def test_save_message_returns_chat_message(
    service: SessionService,
    session_repo: AsyncMock,
    message_repo: AsyncMock,
) -> None:
    """save_message returns the ChatMessageModel from the repository."""
    session_id = uuid.uuid4()
    expected_msg = _make_message(session_id=session_id)
    message_repo.create.return_value = expected_msg
    session_repo.update.return_value = MagicMock()

    result = await service.save_message(
        session_id=session_id,
        user_id=uuid.uuid4(),
        role=MessageRole.assistant,
        content="response",
        tool_calls={"fn": "search"},
        tool_name="search",
        token_usage={"input": 5, "output": 10},
    )

    assert result is expected_msg


# --- get_history ---


@pytest.mark.asyncio
async def test_get_history_delegates_to_message_repo(
    service: SessionService,
    message_repo: AsyncMock,
) -> None:
    """get_history calls get_recent with correct session_id and limit."""
    session_id = uuid.uuid4()
    msgs = [_make_message(session_id=session_id), _make_message(session_id=session_id)]
    message_repo.get_recent.return_value = msgs

    result = await service.get_history(session_id=session_id, limit=10)

    message_repo.get_recent.assert_called_once_with(session_id, 10)
    assert result == msgs


@pytest.mark.asyncio
async def test_get_history_default_limit(
    service: SessionService,
    message_repo: AsyncMock,
) -> None:
    """get_history uses default limit of 50 when not specified."""
    session_id = uuid.uuid4()
    message_repo.get_recent.return_value = []

    await service.get_history(session_id=session_id)

    message_repo.get_recent.assert_called_once_with(session_id, 50)


# --- get_or_create_session ---


@pytest.mark.asyncio
async def test_get_or_create_session_creates_when_none(
    service: SessionService,
    session_repo: AsyncMock,
) -> None:
    """get_or_create_session creates a new session when session_id is None."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    new_session = _make_session(user_id=user_id, agent_id=agent_id)
    session_repo.create.return_value = new_session

    result = await service.get_or_create_session(
        session_id=None,
        user_id=user_id,
        agent_id=agent_id,
    )

    session_repo.create.assert_called_once_with(user_id=user_id, agent_id=agent_id)
    session_repo.get_by_id.assert_not_called()
    assert result is new_session


@pytest.mark.asyncio
async def test_get_or_create_session_returns_existing(
    service: SessionService,
    session_repo: AsyncMock,
) -> None:
    """get_or_create_session returns the existing session when session_id is found."""
    session_id = uuid.uuid4()
    existing = _make_session(session_id=session_id)
    session_repo.get_by_id.return_value = existing

    result = await service.get_or_create_session(
        session_id=session_id,
        user_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
    )

    session_repo.get_by_id.assert_called_once_with(session_id)
    session_repo.create.assert_not_called()
    assert result is existing


@pytest.mark.asyncio
async def test_get_or_create_session_raises_404_when_not_found(
    service: SessionService,
    session_repo: AsyncMock,
) -> None:
    """get_or_create_session raises HTTP 404 when session_id is provided but not found."""
    session_repo.get_by_id.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await service.get_or_create_session(
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
        )

    assert exc_info.value.status_code == 404
