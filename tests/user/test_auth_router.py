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
