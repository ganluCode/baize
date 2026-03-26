"""Tests for the Agent chat SSE endpoint (F-014)."""

import os
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.agent.service import ChatEvent  # noqa: E402
from baize.core.deps import get_agent_service  # noqa: E402
from baize.main import app  # noqa: E402
from baize.user.deps import get_current_user  # noqa: E402
from baize.user.models import UserModel  # noqa: E402

_AGENT_ID = uuid.uuid4()
_SESSION_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()

_CHAT_URL = f"/api/v1/agents/{_AGENT_ID}/sessions/{_SESSION_ID}/chat"


def _make_user() -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = _USER_ID
    return user


async def _async_events(*events: ChatEvent):
    """Async generator yielding the provided ChatEvent objects."""
    for event in events:
        yield event


@pytest.fixture
def mock_agent_svc():
    return MagicMock()


@pytest.fixture
async def client_auth(app, mock_agent_svc):
    """Authenticated client with AgentService mocked."""
    user = _make_user()
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_svc
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def client_no_auth(app, mock_agent_svc):
    """Client with no valid authentication."""
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_svc

    def _raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


async def test_chat_empty_message_returns_422(client_auth, mock_agent_svc):
    response = await client_auth.post(_CHAT_URL, json={"message": ""})
    assert response.status_code == 422


async def test_chat_missing_message_returns_422(client_auth, mock_agent_svc):
    response = await client_auth.post(_CHAT_URL, json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_chat_no_auth_returns_401_json(client_no_auth):
    response = await client_no_auth.post(_CHAT_URL, json={"message": "Hello"})
    assert response.status_code == 401
    # Must be JSON, not SSE
    assert response.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# Successful streaming
# ---------------------------------------------------------------------------


async def test_chat_returns_200_with_sse_content_type(client_auth, mock_agent_svc):
    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hi"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(uuid.uuid4())}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "Hello"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


async def test_chat_response_contains_token_and_done_events(client_auth, mock_agent_svc):
    msg_id = uuid.uuid4()
    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hi there"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(msg_id)}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "Hello"})

    body = response.text
    assert "event: token" in body
    assert "Hi there" in body
    assert "event: done" in body


# ---------------------------------------------------------------------------
# Agent ownership error in SSE stream
# ---------------------------------------------------------------------------


async def test_chat_agent_not_owned_sends_error_event(client_auth, mock_agent_svc):
    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="error", payload={"message": "Agent not found."}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "Hello"})

    # Still returns 200 with SSE stream; the error is in the event payload
    assert response.status_code == 200
    body = response.text
    assert "event: error" in body
    assert "Agent not found." in body


# ---------------------------------------------------------------------------
# Service is called with correct arguments
# ---------------------------------------------------------------------------


async def test_chat_calls_service_with_correct_args(client_auth, mock_agent_svc):
    user = _make_user()
    app.dependency_overrides[get_current_user] = lambda: user

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(uuid.uuid4())}),
        )
    )

    await client_auth.post(_CHAT_URL, json={"message": "Test message"})

    mock_agent_svc.chat.assert_called_once()
    call_kwargs = mock_agent_svc.chat.call_args.kwargs
    assert call_kwargs["agent_id"] == _AGENT_ID
    assert call_kwargs["session_id"] == _SESSION_ID
    assert call_kwargs["message"] == "Test message"
