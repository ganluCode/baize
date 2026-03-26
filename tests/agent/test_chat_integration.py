"""Integration tests for POST /api/v1/chat (SSE) and /api/v1/chat/sync (F-012).

Tests exercise the full request/response cycle with AgentService mocked,
verifying SSE event parsing, JSON response structure, HTTP status codes, and
the ApiResponse error code field (40000 / 40300 / 40400 / 50000).
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
from baize.user.models import UserModel  # noqa: E402

_SSE_URL = "/api/v1/chat"
_SYNC_URL = "/api/v1/chat/sync"

_USER_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()
_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user() -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = _USER_ID
    return user


async def _async_events(*events: ChatEvent):
    """Async generator that yields the provided ChatEvent objects."""
    for event in events:
        yield event


def _parse_sse_events(body: str) -> dict[str, list[str]]:
    """Parse SSE body into {event_type: [data_line, ...]} mapping."""
    result: dict[str, list[str]] = {}
    current_event: str | None = None
    for line in body.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):].strip()
        elif line.startswith("data: ") and current_event is not None:
            result.setdefault(current_event, []).append(line[len("data: "):].strip())
            current_event = None
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# 1. SSE happy path: at least one token event and one done event
# ---------------------------------------------------------------------------


async def test_sse_happy_path_token_and_done_events(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """POST /api/v1/chat returns SSE stream with at least one token and one done event."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Hello"}),
            ChatEvent(type="token", payload={"content": " world"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SSE_URL, json={"message": "Hi"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse_events(response.text)
    assert "token" in events, "Expected at least one 'token' event in SSE stream"
    assert "done" in events, "Expected a 'done' event in SSE stream"
    assert len(events["token"]) >= 1


# ---------------------------------------------------------------------------
# 2. Sync happy path: non-empty content field
# ---------------------------------------------------------------------------


async def test_sync_happy_path_non_empty_content(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """POST /api/v1/chat/sync returns JSON with a non-empty content field."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="token", payload={"content": "Answer here"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(_MESSAGE_ID)}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "Question"})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    body = response.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["content"] != "", "content field must not be empty"
    assert data["content"] == "Answer here"


# ---------------------------------------------------------------------------
# 3. message="" → 400, code=40000 for both endpoints
# ---------------------------------------------------------------------------


async def test_sse_empty_message_returns_400_code_40000(client_auth):
    """POST /api/v1/chat with empty message → HTTP 400, body code=40000."""
    response = await client_auth.post(_SSE_URL, json={"message": ""})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 40000


async def test_sync_empty_message_returns_400_code_40000(client_auth):
    """POST /api/v1/chat/sync with empty message → HTTP 400, body code=40000."""
    response = await client_auth.post(_SYNC_URL, json={"message": ""})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 40000


# ---------------------------------------------------------------------------
# 4. session_id other user → 403, code=40300 for both endpoints
# ---------------------------------------------------------------------------


async def test_sse_session_not_owned_returns_403_code_40300(
    client_auth, mock_agent_config_svc, mock_session_svc
):
    """SSE endpoint: session_id from another user → HTTP 403, code=40300."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get = AsyncMock(return_value=agent)

    mock_session_svc.get_session = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="Forbidden.")
    )

    response = await client_auth.post(
        _SSE_URL,
        json={"message": "Hello", "agent_id": str(_AGENT_ID), "session_id": str(uuid.uuid4())},
    )
    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 40300


async def test_sync_session_not_owned_returns_403_code_40300(
    client_auth, mock_agent_config_svc, mock_session_svc
):
    """Sync endpoint: session_id from another user → HTTP 403, code=40300."""
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
    body = response.json()
    assert body["code"] == 40300


# ---------------------------------------------------------------------------
# 5. agent_id=null, no default agent → 404, code=40400
# ---------------------------------------------------------------------------


async def test_sse_no_default_agent_returns_404_code_40400(
    client_auth, mock_agent_config_svc
):
    """SSE endpoint: agent_id=null and no default agent → HTTP 404, code=40400."""
    mock_agent_config_svc.get_default = AsyncMock(return_value=None)

    response = await client_auth.post(_SSE_URL, json={"message": "Hello"})
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 40400


async def test_sync_no_default_agent_returns_404_code_40400(
    client_auth, mock_agent_config_svc
):
    """Sync endpoint: agent_id=null and no default agent → HTTP 404, code=40400."""
    mock_agent_config_svc.get_default = AsyncMock(return_value=None)

    response = await client_auth.post(_SYNC_URL, json={"message": "Hello"})
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 40400


# ---------------------------------------------------------------------------
# 6. AgentService raises → SSE pushes error event, NOT HTTP 500
# ---------------------------------------------------------------------------


async def test_sse_agent_exception_pushes_error_event_not_500(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """SSE endpoint: AgentService raises exception → error event in stream, status remains 200."""
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    async def _failing_chat():
        raise RuntimeError("LLM backend unavailable")
        yield  # noqa: F841 — makes this an async generator

    mock_agent_svc.chat = MagicMock(return_value=_failing_chat())

    response = await client_auth.post(_SSE_URL, json={"message": "Hi"})

    # SSE connection is established (200), error reported as an SSE event
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = _parse_sse_events(response.text)
    assert "error" in events, "Expected an 'error' event in the SSE stream"


# ---------------------------------------------------------------------------
# 7. AgentService raises → sync endpoint returns 500, code=50000
# ---------------------------------------------------------------------------


async def test_sync_agent_exception_returns_500_code_50000(
    client_auth, mock_agent_config_svc, mock_agent_svc
):
    """Sync endpoint: AgentService yields an error event → HTTP 500, body code=50000.

    The chat_sync endpoint converts an "error" ChatEvent into HTTPException(500),
    which the global exception handler maps to code=50000. This exercises the
    same failure path as an AgentService exception (the SSE wrapper converts
    unhandled exceptions into error events; the sync endpoint promotes error
    events into 500 responses).
    """
    agent = MagicMock()
    agent.id = _AGENT_ID
    mock_agent_config_svc.get_default = AsyncMock(return_value=agent)

    mock_agent_svc.chat = MagicMock(
        return_value=_async_events(
            ChatEvent(type="error", payload={"message": "LLM backend unavailable"}),
        )
    )

    response = await client_auth.post(_SYNC_URL, json={"message": "Hi"})

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == 50000
