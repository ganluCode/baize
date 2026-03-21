"""Tests for POST /api/v1/auth/login, /logout, /refresh endpoints."""

import os
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.main import app  # noqa: E402
from baize.user.auth_router import _get_auth_service  # noqa: E402
from baize.user.auth_service import AuthError, TokenResponse as ServiceTokenResponse  # noqa: E402


@pytest.fixture
def mock_auth():
    return AsyncMock()


@pytest.fixture
async def client(app, mock_auth):
    """Async client with auth_service dependency overridden."""
    app.dependency_overrides[_get_auth_service] = lambda: mock_auth
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(_get_auth_service, None)


async def test_login_valid_credentials_returns_200(client, mock_auth):
    mock_auth.login.return_value = ServiceTokenResponse(
        access_token="tok123",
        token_type="bearer",
        expires_in=86400,
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"login": "ganlu@example.com", "password": "secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "tok123"
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 86400


async def test_login_wrong_password_returns_401(client, mock_auth):
    mock_auth.login.side_effect = AuthError("Invalid credentials.", status_code=401)

    response = await client.post(
        "/api/v1/auth/login",
        json={"login": "ganlu@example.com", "password": "wrong"},
    )

    assert response.status_code == 401


async def test_logout_with_valid_jwt_returns_200(client, mock_auth):
    mock_auth.logout.return_value = None

    response = await client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": "Bearer validtoken"},
    )

    assert response.status_code == 200
    mock_auth.logout.assert_awaited_once_with(token="validtoken")


async def test_logout_without_token_returns_401(client, mock_auth):
    response = await client.post("/api/v1/auth/logout")
    assert response.status_code == 401


async def test_refresh_with_valid_jwt_returns_new_token(client, mock_auth):
    mock_auth.refresh.return_value = ServiceTokenResponse(
        access_token="newtoken",
        token_type="bearer",
        expires_in=86400,
    )

    response = await client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": "Bearer oldtoken"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "newtoken"
    mock_auth.refresh.assert_awaited_once_with(token="oldtoken")


async def test_refresh_without_token_returns_401(client, mock_auth):
    response = await client.post("/api/v1/auth/refresh")
    assert response.status_code == 401


async def test_login_via_name_returns_200(client, mock_auth):
    """Login with username (no @) returns a token."""
    mock_auth.login.return_value = ServiceTokenResponse(
        access_token="tok_name",
        token_type="bearer",
        expires_in=86400,
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"login": "ganlu", "password": "secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "tok_name"
    mock_auth.login.assert_awaited_once_with(login="ganlu", password="secret")


async def test_login_inactive_user_returns_403(client, mock_auth):
    """is_active=False user is rejected with 403."""
    mock_auth.login.side_effect = AuthError("User account is inactive.", status_code=403)

    response = await client.post(
        "/api/v1/auth/login",
        json={"login": "ganlu@example.com", "password": "secret"},
    )

    assert response.status_code == 403


async def test_logout_then_token_becomes_invalid(client, mock_auth):
    """After logout, using the same token again returns 401."""
    # First call: token is valid, logout succeeds
    # Second call: token is revoked, logout raises (simulating revoked JWT)
    mock_auth.logout.side_effect = [None, Exception("Token already revoked")]

    resp1 = await client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": "Bearer mytoken"},
    )
    assert resp1.status_code == 200

    resp2 = await client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": "Bearer mytoken"},
    )
    assert resp2.status_code == 401


async def test_refresh_then_old_token_becomes_invalid(client, mock_auth):
    """After refresh, using the old token again returns 401."""
    mock_auth.refresh.side_effect = [
        ServiceTokenResponse(access_token="newtoken", token_type="bearer", expires_in=86400),
        Exception("Token already revoked"),
    ]

    resp1 = await client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": "Bearer oldtoken"},
    )
    assert resp1.status_code == 200
    assert resp1.json()["access_token"] == "newtoken"

    # Old token is now revoked — second refresh with same token fails
    resp2 = await client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": "Bearer oldtoken"},
    )
    assert resp2.status_code == 401
