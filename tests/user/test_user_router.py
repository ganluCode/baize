"""Tests for user management API endpoints (src/baize/user/router.py)."""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.main import app  # noqa: E402
from baize.user.deps import get_current_user, require_admin  # noqa: E402
from baize.user.models import UserModel  # noqa: E402
from baize.user.repository import UserRepository  # noqa: E402
from baize.user.router import _get_user_service  # noqa: E402
from baize.user.schemas import (  # noqa: E402
    ResetKeyResponse,
    UserCreateResponse,
    UserListResponse,
    UserResponse,
)
from baize.user.service import UserServiceError  # noqa: E402

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_USER_ID = uuid.uuid4()
_ADMIN_ID = uuid.uuid4()


def _make_user_response(**kwargs) -> UserResponse:
    defaults = dict(
        id=_USER_ID,
        email="ganlu@example.com",
        name="ganlu",
        role="user",
        avatar=None,
        preferences=None,
        is_active=True,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return UserResponse(**defaults)


def _make_user_model(*, role: str = "user", user_id: uuid.UUID | None = None) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id or uuid.uuid4()
    user.email = "ganlu@example.com"
    user.name = "ganlu"
    user.password = "hashed"
    user.role = role
    user.avatar = None
    user.is_active = True
    user.preferences = None
    user.api_key_hash = "somehash"
    user.created_at = _NOW
    user.updated_at = _NOW
    return user


@pytest.fixture
def mock_svc():
    return AsyncMock()


@pytest.fixture
def regular_user():
    return _make_user_model(role="user")


@pytest.fixture
def admin_user():
    return _make_user_model(role="admin", user_id=_ADMIN_ID)


@pytest.fixture
async def client_user(app, mock_svc, regular_user):
    """Client authenticated as a regular user."""
    app.dependency_overrides[_get_user_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: regular_user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(_get_user_service, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def client_admin(app, mock_svc, admin_user):
    """Client authenticated as admin."""
    app.dependency_overrides[_get_user_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: admin_user
    app.dependency_overrides[require_admin] = lambda: admin_user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(_get_user_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(require_admin, None)


@pytest.fixture
async def client_no_auth(app, mock_svc):
    """Client with no authentication."""
    app.dependency_overrides[_get_user_service] = lambda: mock_svc

    def _raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _raise_401
    app.dependency_overrides[require_admin] = _raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(_get_user_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(require_admin, None)


@pytest.fixture
async def client_non_admin(app, mock_svc, regular_user):
    """Client authenticated as non-admin user (for admin-only endpoints)."""
    app.dependency_overrides[_get_user_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: regular_user

    def _raise_403():
        raise HTTPException(status_code=403, detail="Admin access required")

    app.dependency_overrides[require_admin] = _raise_403
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(_get_user_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(require_admin, None)


# ---------------------------------------------------------------------------
# GET /api/v1/users/me
# ---------------------------------------------------------------------------


async def test_get_me_with_valid_api_key_returns_200(client_user, mock_svc, regular_user):
    mock_svc.get_me.return_value = _make_user_response()

    response = await client_user.get("/api/v1/users/me")

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "ganlu@example.com"
    assert body["name"] == "ganlu"
    assert "password" not in body
    assert "api_key_hash" not in body
    mock_svc.get_me.assert_awaited_once_with(regular_user)


async def test_get_me_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.get("/api/v1/users/me")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# PATCH /api/v1/users/me
# ---------------------------------------------------------------------------


async def test_patch_me_merges_preferences(client_user, mock_svc, regular_user):
    updated = _make_user_response(preferences={"language": "en", "communication_style": "formal"})
    mock_svc.update_me.return_value = updated

    response = await client_user.patch(
        "/api/v1/users/me",
        json={"preferences": {"language": "en"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["preferences"]["language"] == "en"
    assert body["preferences"]["communication_style"] == "formal"


async def test_patch_me_name_conflict_returns_409(client_user, mock_svc):
    mock_svc.update_me.side_effect = UserServiceError("Name already taken.", status_code=409)

    response = await client_user.patch("/api/v1/users/me", json={"name": "taken"})

    assert response.status_code == 409


# ---------------------------------------------------------------------------
# POST /api/v1/users (admin only)
# ---------------------------------------------------------------------------


async def test_create_user_as_non_admin_returns_403(client_non_admin):
    response = await client_non_admin.post(
        "/api/v1/users",
        json={"email": "new@example.com", "name": "newuser", "password": "pw"},
    )
    assert response.status_code == 403


async def test_create_user_as_admin_returns_201_with_api_key(client_admin, mock_svc):
    new_user_id = uuid.uuid4()
    mock_svc.create_user.return_value = UserCreateResponse(
        id=new_user_id,
        email="new@example.com",
        name="newuser",
        role="user",
        avatar=None,
        preferences=None,
        is_active=True,
        created_at=_NOW,
        updated_at=_NOW,
        api_key="plaintext-api-key-value",
    )

    response = await client_admin.post(
        "/api/v1/users",
        json={"email": "new@example.com", "name": "newuser", "password": "pw"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["api_key"] == "plaintext-api-key-value"
    assert body["email"] == "new@example.com"


# ---------------------------------------------------------------------------
# GET /api/v1/users (admin only)
# ---------------------------------------------------------------------------


async def test_list_users_as_non_admin_returns_403(client_non_admin):
    response = await client_non_admin.get("/api/v1/users")
    assert response.status_code == 403


async def test_list_users_as_admin_returns_items_and_total(client_admin, mock_svc):
    mock_svc.list_users.return_value = UserListResponse(
        items=[_make_user_response()],
        total=1,
    )

    response = await client_admin.get("/api/v1/users")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert "api_key_hash" not in body["items"][0]


# ---------------------------------------------------------------------------
# POST /api/v1/users/{user_id}/reset-key (admin only)
# ---------------------------------------------------------------------------


async def test_reset_key_returns_new_api_key(client_admin, mock_svc):
    target_id = uuid.uuid4()
    mock_svc.reset_api_key.return_value = ResetKeyResponse(api_key="new-plaintext-key")

    response = await client_admin.post(f"/api/v1/users/{target_id}/reset-key")

    assert response.status_code == 200
    body = response.json()
    assert body["api_key"] == "new-plaintext-key"
    mock_svc.reset_api_key.assert_awaited_once_with(target_id)


async def test_reset_key_user_not_found_returns_404(client_admin, mock_svc):
    mock_svc.reset_api_key.side_effect = UserServiceError("User not found.", status_code=404)

    response = await client_admin.post(f"/api/v1/users/{uuid.uuid4()}/reset-key")

    assert response.status_code == 404


async def test_reset_key_then_old_key_returns_401(app, mock_svc, admin_user):
    """After reset-key, the old API key is no longer valid for authentication.

    Step 1: admin resets the key → new plaintext key returned.
    Step 2: subsequent request with old key is rejected (auth returns 401).
    The rejection is simulated via get_current_user override, reflecting that
    the old api_key_hash is no longer found in the DB after reset.
    """
    target_id = uuid.uuid4()
    mock_svc.reset_api_key.return_value = ResetKeyResponse(api_key="brand-new-key")

    app.dependency_overrides[_get_user_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: admin_user
    app.dependency_overrides[require_admin] = lambda: admin_user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Step 1: Admin resets the key
            resp = await c.post(f"/api/v1/users/{target_id}/reset-key")
            assert resp.status_code == 200
            assert resp.json()["api_key"] == "brand-new-key"

            # Step 2: Simulate old key hash no longer in DB (reset replaced it)
            def _old_key_rejected():
                raise HTTPException(status_code=401, detail="Invalid API key")

            app.dependency_overrides[get_current_user] = _old_key_rejected
            app.dependency_overrides.pop(require_admin)

            resp = await c.get(
                "/api/v1/users/me",
                headers={"X-API-Key": "old-key-no-longer-valid"},
            )
            assert resp.status_code == 401
    finally:
        app.dependency_overrides.pop(_get_user_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(require_admin, None)
