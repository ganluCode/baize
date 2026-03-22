"""Tests for POST /api/v1/chat streaming SSE endpoint (F-009).

Covers:
- Message validation (empty → 400)
- Default agent resolution (agent_id=null → default, no default → 404)
- Session auto-creation (session_id=null → new session in done event)
- Session ownership (session_id not owned → 403)
- SSE event format (token, tool_call, tool_result, done, error)
- Auth: uses auth.deps.get_current_user (Service Key → 403)
- Agent exception → SSE error event, not HTTP 500
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.agent.chat_router import _get_user_model  # noqa: E402
from baize.agent.service import ChatEvent  # noqa: E402
from baize.auth.deps import get_current_user as auth_get_current_user  # noqa: E402
from baize.core.deps import (  # noqa: E402
    get_agent_config_service,
    get_agent_service,
    get_session_service,
)
from baize.main import app  # noqa: E402
from baize.user.models import UserModel  # noqa: E402

_CHAT_URL = "/api/v1/chat"
_USER_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()
_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()


def _make_user() -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = _USER_ID
    return user


async def _async_events(*events: ChatEvent):
    for event in events:
        yield event


@pytest.fixture
def mock_agent_svc():
    return MagicMock()


@pytest.fixture
def mock_agent_config_svc():
    return MagicMock()


@pytest.fixture
def mock_session_svc():
    return MagicMock()


@pytest.fixture
async def client_auth(app, mock_agent_svc, mock_agent_config_svc, mock_session_svc):
    """Authenticated client with all services mocked."""
    user = _make_user()
    app.dependency_overrides[_get_user_model] = lambda: user
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_svc
    app.dependency_overrides[get_agent_config_service] = lambda: mock_agent_config_svc
    app.dependency_overrides[get_session_service] = lambda: mock_session_svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        for dep in [_get_user_model, get_agent_service, get_agent_config_service, get_session_service]:
            app.dependency_overrides.pop(dep, None)


# ---------------------------------------------------------------------------
# Message validation
# ---------------------------------------------------------------------------


async def test_empty_message_returns_400(client_auth):
    response = await client_auth.post(_CHAT_URL, json={"message": ""})
    assert response.status_code == 400
    assert response.json()["message"] == "Message cannot be empty"


# ---------------------------------------------------------------------------
# Default agent resolution
# ---------------------------------------------------------------------------


async def test_null_agent_id_uses_default_agent(
    client_auth, mock_agent_config_svc, mock_agent_svc,
):
    """agent_id=null resolves to the user's default agent."""
    default_agent = MagicMock()
    default_agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=default_agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(
        _CHAT_URL, json={"message": "Hello", "agent_id": None}
    )
    assert response.status_code == 200
    mock_agent_config_svc.get_default.assert_awaited_once()


async def test_null_agent_id_no_default_returns_404(
    client_auth, mock_agent_config_svc
):
    """agent_id=null and user has no default agent → 404."""
    mock_agent_config_svc.get_default = AsyncMock(return_value=None)

    response = await client_auth.post(
        _CHAT_URL, json={"message": "Hello"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Session auto-creation
# ---------------------------------------------------------------------------


async def test_null_session_id_creates_new_session(
    client_auth, mock_agent_config_svc, mock_agent_svc,
):
    """session_id=null auto-creates a new session; done event contains new session_id."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    new_session_id = uuid.uuid4()
    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="done", payload={"session_id": str(new_session_id), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(
        _CHAT_URL, json={"message": "Hello"}
    )
    assert response.status_code == 200
    body = response.text
    assert "event: done" in body
    assert str(new_session_id) in body


# ---------------------------------------------------------------------------
# Session ownership
# ---------------------------------------------------------------------------


async def test_session_not_owned_returns_403(
    client_auth, mock_agent_config_svc, mock_session_svc
):
    """session_id belonging to another user → 403."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get = AsyncMock(return_value=agent)

    mock_session_svc.get_session = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="Forbidden.")
    )

    other_session_id = str(uuid.uuid4())
    response = await client_auth.post(
        _CHAT_URL,
        json={"message": "Hello", "agent_id": str(_AGENT_ID), "session_id": other_session_id},
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# SSE event format
# ---------------------------------------------------------------------------


async def test_sse_response_content_type(
    client_auth, mock_agent_config_svc, mock_agent_svc,
):
    """Successful request returns 200 with Content-Type: text/event-stream."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hi"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "Hello"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


async def test_sse_token_events(
    client_auth, mock_agent_config_svc, mock_agent_svc,
):
    """Response stream contains token events with content."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hello world"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "Hi"})
    body = response.text
    assert "event: token" in body
    assert "Hello world" in body


async def test_sse_tool_call_and_tool_result_events(
    client_auth, mock_agent_config_svc, mock_agent_svc,
):
    """Tool invocations produce tool_call and tool_result events."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="tool_call", payload={"tool": "search_memory", "args": {"query": "test"}}),
            ChatEvent(type="tool_result", payload={"tool": "search_memory", "result": "found it"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "search"})
    body = response.text
    assert "event: tool_call" in body
    assert "event: tool_result" in body


async def test_sse_done_event_contains_session_and_message_id(
    client_auth, mock_agent_config_svc, mock_agent_svc,
):
    """Done event data contains session_id and message_id."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    msg_id = uuid.uuid4()
    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(msg_id)}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "Hi"})
    body = response.text
    assert "event: done" in body
    assert str(_SESSION_ID) in body
    assert str(msg_id) in body


# ---------------------------------------------------------------------------
# Agent error → SSE error event
# ---------------------------------------------------------------------------


async def test_agent_exception_sends_sse_error_event(
    client_auth, mock_agent_config_svc, mock_agent_svc,
):
    """When AgentService.chat() raises, the endpoint pushes an error event and closes."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    async def _failing_chat():
        raise RuntimeError("LLM connection failed")
        yield  # noqa: unreachable — makes this an async generator

    mock_agent_svc.chat = MagicMock(return_value=_failing_chat())

    response = await client_auth.post(_CHAT_URL, json={"message": "Hi"})
    # Should still return 200 SSE, with error event in stream
    assert response.status_code == 200
    body = response.text
    assert "event: error" in body


# ---------------------------------------------------------------------------
# Auth: Service Key not allowed
# ---------------------------------------------------------------------------


async def test_service_key_returns_403(app, mock_agent_svc):
    """Service Key auth triggers 403 from get_current_user (auth.deps)."""
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_svc

    def _service_key_user():
        raise HTTPException(status_code=403, detail="User context required")

    app.dependency_overrides[_get_user_model] = _service_key_user

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(_CHAT_URL, json={"message": "Hello"})
    finally:
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(_get_user_model, None)

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Explicit agent_id and session_id
# ---------------------------------------------------------------------------


async def test_explicit_agent_and_session_ids(
    client_auth, mock_agent_config_svc, mock_agent_svc, mock_session_svc
):
    """When both agent_id and session_id are provided, they are used directly."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get = AsyncMock(return_value=agent)

    session = MagicMock()
    session.id = _SESSION_ID
    session.user_id = _USER_ID
    mock_session_svc.get_session = AsyncMock(return_value=session)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(
        _CHAT_URL,
        json={"message": "Hello", "agent_id": str(_AGENT_ID), "session_id": str(_SESSION_ID)},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
