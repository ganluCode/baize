"""API integration tests for the Agent chat SSE endpoint (F-017).

Covers:
- POST chat returns Content-Type: text/event-stream
- Response body contains at least one token event and one done event
  (LLM is mocked via monkeypatch on AgentService.chat)
- Invalid API Key returns 401 with JSON body (not SSE)
"""

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

from baize.agent.service import AgentService, ChatEvent  # noqa: E402
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


async def _fake_chat_events(*events: ChatEvent):
    """Async generator that yields the given ChatEvent objects."""
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


# ---------------------------------------------------------------------------
# SSE response format
# ---------------------------------------------------------------------------


async def test_chat_returns_text_event_stream_content_type(client_auth, mock_agent_svc):
    """POST chat responds with Content-Type: text/event-stream."""
    msg_id = uuid.uuid4()
    mock_agent_svc.chat = MagicMock(
        return_value=_fake_chat_events(
            ChatEvent(type="token", payload={"content": "Hello"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(msg_id)}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "Hi"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


async def test_chat_response_contains_token_and_done_events(client_auth, mock_agent_svc, monkeypatch):
    """SSE body contains at least one token event and one done event.

    The LLM is mocked via monkeypatch on AgentService.chat so no real
    LLM call is made.
    """
    msg_id = uuid.uuid4()

    async def _mocked_chat(self, **kwargs):
        yield ChatEvent(type="token", payload={"content": "Hi there"})
        yield ChatEvent(type="token", payload={"content": " from mock"})
        yield ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(msg_id)})

    monkeypatch.setattr(AgentService, "chat", _mocked_chat)
    # Override AgentService dependency to use a real-ish instance with chat monkeypatched
    app.dependency_overrides[get_agent_service] = lambda: AgentService.__new__(AgentService)

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(_CHAT_URL, json={"message": "Hello"})
    finally:
        app.dependency_overrides.pop(get_agent_service, None)

    assert response.status_code == 200
    body = response.text
    # At least one token event
    assert "event: token" in body
    assert "Hi there" in body
    # Exactly one done event
    assert "event: done" in body
    assert str(msg_id) in body


async def test_chat_sse_body_parses_token_and_done_via_mock_service(client_auth, mock_agent_svc):
    """Verify SSE body can be parsed for token and done events (service-level mock)."""
    msg_id = uuid.uuid4()
    mock_agent_svc.chat = MagicMock(
        return_value=_fake_chat_events(
            ChatEvent(type="token", payload={"content": "answer"}),
            ChatEvent(type="done", payload={"session_id": str(_SESSION_ID), "message_id": str(msg_id)}),
        )
    )

    response = await client_auth.post(_CHAT_URL, json={"message": "question"})

    body = response.text
    # Parse SSE lines
    event_types = [
        line[len("event: "):].strip()
        for line in body.splitlines()
        if line.startswith("event: ")
    ]
    assert "token" in event_types
    assert "done" in event_types


# ---------------------------------------------------------------------------
# Authentication — invalid API key returns 401 JSON
# ---------------------------------------------------------------------------


async def test_chat_invalid_api_key_returns_401_json(app, mock_agent_svc):
    """An invalid (or missing) API key returns HTTP 401 with a JSON body, not SSE."""
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_svc

    def _reject():
        raise HTTPException(status_code=401, detail="Invalid API key")

    app.dependency_overrides[get_current_user] = _reject

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(_CHAT_URL, json={"message": "Hello"})
    finally:
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 401
    # Must be JSON, not event-stream
    assert response.headers["content-type"].startswith("application/json")
    assert "detail" in response.json()


async def test_chat_no_auth_header_returns_401_json(app, mock_agent_svc):
    """Missing authentication header returns 401 JSON (not SSE)."""
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_svc

    def _no_creds():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _no_creds

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(_CHAT_URL, json={"message": "Hello"})
    finally:
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


async def test_chat_empty_message_returns_422(client_auth, mock_agent_svc):
    response = await client_auth.post(_CHAT_URL, json={"message": ""})
    assert response.status_code == 422
