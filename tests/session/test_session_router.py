"""Tests for session create and list API endpoints (F-007)."""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.main import app  # noqa: E402
from baize.session.schemas import SessionResponse  # noqa: E402
from baize.session.models import SessionStatus  # noqa: E402
from baize.user.deps import get_current_user  # noqa: E402
from baize.user.models import UserModel  # noqa: E402

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_USER_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()
_SESSION_ID = uuid.uuid4()


def _make_user_model() -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = _USER_ID
    return user


def _make_session_response(**kwargs) -> SessionResponse:
    defaults = dict(
        id=_SESSION_ID,
        user_id=_USER_ID,
        agent_id=_AGENT_ID,
        title=None,
        status=SessionStatus.active,
        auto_memory_recall=None,
        shared_memory=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return SessionResponse(**defaults)


@pytest.fixture
def mock_svc():
    return AsyncMock()


@pytest.fixture
def regular_user():
    return _make_user_model()


@pytest.fixture
async def client_auth(app, mock_svc, regular_user):
    """Client authenticated as a regular user with session service mocked."""
    from baize.core.deps import get_session_service

    app.dependency_overrides[get_session_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: regular_user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def client_no_auth(app, mock_svc):
    """Client with no authentication."""
    from baize.core.deps import get_session_service

    app.dependency_overrides[get_session_service] = lambda: mock_svc

    def _raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# POST /api/v1/agents/{agent_id}/sessions
# ---------------------------------------------------------------------------


async def test_create_session_with_title_returns_201(client_auth, mock_svc):
    mock_svc.create_session.return_value = _make_session_response(title="My Session")

    response = await client_auth.post(
        f"/api/v1/agents/{_AGENT_ID}/sessions",
        json={"title": "My Session"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "My Session"
    assert body["agent_id"] == str(_AGENT_ID)
    assert body["user_id"] == str(_USER_ID)
    assert body["status"] == "active"


async def test_create_session_without_title_returns_null_title(client_auth, mock_svc):
    mock_svc.create_session.return_value = _make_session_response(title=None)

    response = await client_auth.post(
        f"/api/v1/agents/{_AGENT_ID}/sessions",
        json={},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] is None


async def test_create_session_with_null_title_returns_null_title(client_auth, mock_svc):
    mock_svc.create_session.return_value = _make_session_response(title=None)

    response = await client_auth.post(
        f"/api/v1/agents/{_AGENT_ID}/sessions",
        json={"title": None},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] is None


async def test_create_session_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.post(
        f"/api/v1/agents/{_AGENT_ID}/sessions",
        json={},
    )
    assert response.status_code == 401


async def test_create_session_calls_service_with_correct_args(client_auth, mock_svc, regular_user):
    mock_svc.create_session.return_value = _make_session_response(title="T")

    await client_auth.post(
        f"/api/v1/agents/{_AGENT_ID}/sessions",
        json={"title": "T"},
    )

    mock_svc.create_session.assert_awaited_once_with(
        user_id=regular_user.id,
        agent_id=_AGENT_ID,
        title="T",
    )


# ---------------------------------------------------------------------------
# GET /api/v1/agents/{agent_id}/sessions
# ---------------------------------------------------------------------------


async def test_list_sessions_returns_200_with_schema(client_auth, mock_svc):
    mock_svc.list_sessions.return_value = ([_make_session_response()], 1)

    response = await client_auth.get(f"/api/v1/agents/{_AGENT_ID}/sessions")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == str(_SESSION_ID)


async def test_list_sessions_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.get(f"/api/v1/agents/{_AGENT_ID}/sessions")
    assert response.status_code == 401


async def test_list_sessions_default_pagination(client_auth, mock_svc, regular_user):
    mock_svc.list_sessions.return_value = ([], 0)

    await client_auth.get(f"/api/v1/agents/{_AGENT_ID}/sessions")

    mock_svc.list_sessions.assert_awaited_once_with(
        user_id=regular_user.id,
        agent_id=_AGENT_ID,
        status=None,
        limit=20,
        offset=0,
    )


async def test_list_sessions_with_status_filter(client_auth, mock_svc, regular_user):
    mock_svc.list_sessions.return_value = ([], 0)

    await client_auth.get(f"/api/v1/agents/{_AGENT_ID}/sessions?status=active")

    call_kwargs = mock_svc.list_sessions.call_args
    assert call_kwargs.kwargs["status"] == SessionStatus.active


async def test_list_sessions_with_pagination_params(client_auth, mock_svc, regular_user):
    mock_svc.list_sessions.return_value = ([], 0)

    await client_auth.get(f"/api/v1/agents/{_AGENT_ID}/sessions?limit=5&offset=10")

    call_kwargs = mock_svc.list_sessions.call_args
    assert call_kwargs.kwargs["limit"] == 5
    assert call_kwargs.kwargs["offset"] == 10


async def test_list_sessions_only_returns_current_user_data(client_auth, mock_svc, regular_user):
    """The service is called with the authenticated user's id."""
    mock_svc.list_sessions.return_value = ([], 0)

    await client_auth.get(f"/api/v1/agents/{_AGENT_ID}/sessions")

    call_kwargs = mock_svc.list_sessions.call_args
    assert call_kwargs.kwargs["user_id"] == regular_user.id


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_get_session_returns_200_when_found(client_auth, mock_svc):
    mock_svc.get_session.return_value = _make_session_response()

    response = await client_auth.get(f"/api/v1/sessions/{_SESSION_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(_SESSION_ID)
    assert body["user_id"] == str(_USER_ID)


async def test_get_session_not_found_returns_404(client_auth, mock_svc):
    from fastapi import HTTPException

    mock_svc.get_session.side_effect = HTTPException(status_code=404, detail="Session not found.")

    response = await client_auth.get(f"/api/v1/sessions/{_SESSION_ID}")

    assert response.status_code == 404


async def test_get_session_wrong_owner_returns_403(client_auth, mock_svc):
    from fastapi import HTTPException

    mock_svc.get_session.side_effect = HTTPException(status_code=403, detail="Forbidden.")

    response = await client_auth.get(f"/api/v1/sessions/{_SESSION_ID}")

    assert response.status_code == 403


async def test_get_session_calls_service_with_correct_args(client_auth, mock_svc, regular_user):
    mock_svc.get_session.return_value = _make_session_response()

    await client_auth.get(f"/api/v1/sessions/{_SESSION_ID}")

    mock_svc.get_session.assert_awaited_once_with(
        session_id=_SESSION_ID,
        user_id=regular_user.id,
    )


# ---------------------------------------------------------------------------
# PATCH /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_archive_session_returns_200_with_archived_status(client_auth, mock_svc):
    mock_svc.archive_session.return_value = _make_session_response(status=SessionStatus.archived)

    response = await client_auth.patch(
        f"/api/v1/sessions/{_SESSION_ID}",
        json={"status": "archived"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "archived"


async def test_archive_session_invalid_status_returns_422(client_auth, mock_svc):
    response = await client_auth.patch(
        f"/api/v1/sessions/{_SESSION_ID}",
        json={"status": "active"},
    )

    assert response.status_code == 422


async def test_archive_session_wrong_owner_returns_403(client_auth, mock_svc):
    from fastapi import HTTPException

    mock_svc.archive_session.side_effect = HTTPException(status_code=403, detail="Forbidden.")

    response = await client_auth.patch(
        f"/api/v1/sessions/{_SESSION_ID}",
        json={"status": "archived"},
    )

    assert response.status_code == 403


async def test_archive_session_calls_service_with_correct_args(client_auth, mock_svc, regular_user):
    mock_svc.archive_session.return_value = _make_session_response(status=SessionStatus.archived)

    await client_auth.patch(
        f"/api/v1/sessions/{_SESSION_ID}",
        json={"status": "archived"},
    )

    mock_svc.archive_session.assert_awaited_once_with(
        session_id=_SESSION_ID,
        user_id=regular_user.id,
    )


# ---------------------------------------------------------------------------
# DELETE /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_delete_session_returns_204(client_auth, mock_svc):
    mock_svc.delete_session.return_value = None

    response = await client_auth.delete(f"/api/v1/sessions/{_SESSION_ID}")

    assert response.status_code == 204


async def test_delete_session_wrong_owner_returns_403(client_auth, mock_svc):
    from fastapi import HTTPException

    mock_svc.delete_session.side_effect = HTTPException(status_code=403, detail="Forbidden.")

    response = await client_auth.delete(f"/api/v1/sessions/{_SESSION_ID}")

    assert response.status_code == 403


async def test_delete_session_calls_service_with_correct_args(client_auth, mock_svc, regular_user):
    mock_svc.delete_session.return_value = None

    await client_auth.delete(f"/api/v1/sessions/{_SESSION_ID}")

    mock_svc.delete_session.assert_awaited_once_with(
        session_id=_SESSION_ID,
        user_id=regular_user.id,
    )
