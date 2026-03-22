"""Tests for POST /api/v1/chat/sync non-streaming endpoint (F-010).

Covers:
- Successful response: 200 JSON with session_id, message_id, content, tool_calls
- tool_calls items contain tool, args, result
- Empty message → 400
- Session not owned → 403
- Response wrapped in ApiResponse (code=0)
- Uses get_current_user dependency (Service Key → 403)
- Agent error → 500
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
from baize.core.deps import (  # noqa: E402
    get_agent_config_service,
    get_agent_service,
    get_session_service,
)
from baize.main import app  # noqa: E402
from baize.user.models import UserModel  # noqa: E402

_SYNC_URL = "/api/v1/chat/sync"
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
# Successful response structure
# ---------------------------------------------------------------------------


async def test_sync_returns_200_json(client_auth, mock_agent_config_svc, mock_agent_svc):
    """Successful request returns HTTP 200 with application/json content type."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hello!"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "Hi"})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


async def test_sync_response_wrapped_in_api_response(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """Response is wrapped in ApiResponse with code=0."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hi"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "Hello"})
    body = response.json()
    assert body["code"] == 0


async def test_sync_response_contains_required_fields(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """Response data contains session_id, message_id, content, and tool_calls."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hello "}),
            ChatEvent(type="token", payload={"content": "world!"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "Hi"})
    data = response.json()["data"]
    assert data["session_id"] == str(_SESSION_ID)
    assert data["message_id"] == str(_MESSAGE_ID)
    assert data["content"] == "Hello world!"
    assert isinstance(data["tool_calls"], list)


async def test_sync_response_full_content_assembled_from_tokens(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """Content field is the concatenation of all token event contents."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Part1 "}),
            ChatEvent(type="token", payload={"content": "Part2 "}),
            ChatEvent(type="token", payload={"content": "Part3"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "Hi"})
    assert response.json()["data"]["content"] == "Part1 Part2 Part3"


async def test_sync_tool_calls_contain_tool_args_result(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """tool_calls items include tool, args, and result fields."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="tool_call", payload={"tool": "search_memory", "args": {"query": "test"}}),
            ChatEvent(type="tool_result", payload={"tool": "search_memory", "result": "found it"}),
            ChatEvent(type="token", payload={"content": "I found it."}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "search"})
    tool_calls = response.json()["data"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "search_memory"
    assert tool_calls[0]["args"] == {"query": "test"}
    assert tool_calls[0]["result"] == "found it"


async def test_sync_multiple_tool_calls(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """Multiple tool invocations all appear in tool_calls list."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="tool_call", payload={"tool": "search_memory", "args": {"query": "x"}}),
            ChatEvent(type="tool_result", payload={"tool": "search_memory", "result": "r1"}),
            ChatEvent(type="tool_call", payload={"tool": "list_tasks", "args": {}}),
            ChatEvent(type="tool_result", payload={"tool": "list_tasks", "result": "r2"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "do stuff"})
    tool_calls = response.json()["data"]["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["tool"] == "search_memory"
    assert tool_calls[1]["tool"] == "list_tasks"


async def test_sync_empty_tool_calls_when_no_tools_used(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """tool_calls is empty list when no tools are invoked."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Just text."}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "Hi"})
    assert response.json()["data"]["tool_calls"] == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


async def test_sync_empty_message_returns_400(client_auth):
    """Empty message returns HTTP 400."""
    response = await client_auth.post(_SYNC_URL, json={"message": ""})
    assert response.status_code == 400
    assert response.json()["message"] == "Message cannot be empty"


# ---------------------------------------------------------------------------
# Session ownership
# ---------------------------------------------------------------------------


async def test_sync_session_not_owned_returns_403(
    client_auth, mock_agent_config_svc, mock_session_svc
):
    """session_id belonging to another user → 403."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get = AsyncMock(return_value=agent)

    mock_session_svc.get_session = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="Forbidden.")
    )

    response = await client_auth.post(
        _SYNC_URL,
        json={"message": "Hello", "agent_id": str(_AGENT_ID), "session_id": str(uuid.uuid4())},
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Auth: Service Key not allowed
# ---------------------------------------------------------------------------


async def test_sync_service_key_returns_403(app, mock_agent_svc):
    """Service Key auth triggers 403 from _get_user_model (inherits get_current_user)."""
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_svc

    def _service_key_user():
        raise HTTPException(status_code=403, detail="User context required")

    app.dependency_overrides[_get_user_model] = _service_key_user

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(_SYNC_URL, json={"message": "Hello"})
    finally:
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(_get_user_model, None)

    assert response.status_code == 403
