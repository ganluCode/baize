"""Unit tests for authentication orchestration dependency (F-007)."""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from fastapi import HTTPException  # noqa: E402

from baize.auth.deps import get_auth_result, get_current_user  # noqa: E402
from baize.auth.schemas import AuthResult  # noqa: E402


def _make_container(secret_key: str = "test-secret") -> MagicMock:
    """Build a minimal mock Container with config and redis."""
    container = MagicMock()
    container.config.secret_key = secret_key
    container.redis = AsyncMock()
    return container


def _make_db() -> AsyncMock:
    return AsyncMock()


# ---------------------------------------------------------------------------
# get_auth_result tests
# ---------------------------------------------------------------------------


class TestGetAuthResult:
    """Tests for the auth orchestration dependency."""

    @pytest.mark.asyncio
    async def test_service_key_match_returns_service_auth_result(self):
        """Service key matching Bearer token returns AuthResult with user_id=None."""
        service_result = AuthResult(user_id=None, auth_type="service")
        container = _make_container()
        db = _make_db()

        with patch("baize.auth.deps.authenticate_service_key", return_value=service_result):
            result = await get_auth_result(
                authorization="Bearer service-secret",
                db=db,
                container=container,
            )

        assert result is service_result
        assert result.user_id is None
        assert result.auth_type == "service"

    @pytest.mark.asyncio
    async def test_valid_jwt_returns_user_auth_result(self):
        """Valid JWT (service key fails) returns AuthResult with non-empty user_id."""
        user_id = str(uuid.uuid4())
        jwt_result = AuthResult(user_id=user_id, auth_type="jwt")
        container = _make_container()
        db = _make_db()

        with (
            patch("baize.auth.deps.authenticate_service_key", return_value=None),
            patch("baize.auth.deps.authenticate_jwt", new=AsyncMock(return_value=jwt_result)),
        ):
            result = await get_auth_result(
                authorization="Bearer some-jwt-token",
                db=db,
                container=container,
            )

        assert result is jwt_result
        assert result.user_id == user_id
        assert result.auth_type == "jwt"

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_user_auth_result(self):
        """Valid API key (service key and JWT fail) returns AuthResult with non-empty user_id."""
        user_id = str(uuid.uuid4())
        api_key_result = AuthResult(user_id=user_id, auth_type="api_key")
        container = _make_container()
        db = _make_db()

        with (
            patch("baize.auth.deps.authenticate_service_key", return_value=None),
            patch("baize.auth.deps.authenticate_jwt", new=AsyncMock(return_value=None)),
            patch("baize.auth.deps.authenticate_api_key", new=AsyncMock(return_value=api_key_result)),
        ):
            result = await get_auth_result(
                authorization="Bearer some-api-key",
                db=db,
                container=container,
            )

        assert result is api_key_result
        assert result.user_id == user_id
        assert result.auth_type == "api_key"

    @pytest.mark.asyncio
    async def test_missing_authorization_header_raises_401_missing_credentials(self):
        """No Authorization header raises HTTP 401 with detail='Missing credentials'."""
        container = _make_container()
        db = _make_db()

        with pytest.raises(HTTPException) as exc_info:
            await get_auth_result(authorization=None, db=db, container=container)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Missing credentials"

    @pytest.mark.asyncio
    async def test_all_authenticators_fail_raises_401_invalid_credentials(self):
        """All three authenticators returning None raises HTTP 401 with detail='Invalid credentials'."""
        container = _make_container()
        db = _make_db()

        with (
            patch("baize.auth.deps.authenticate_service_key", return_value=None),
            patch("baize.auth.deps.authenticate_jwt", new=AsyncMock(return_value=None)),
            patch("baize.auth.deps.authenticate_api_key", new=AsyncMock(return_value=None)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_auth_result(
                    authorization="Bearer bad-token",
                    db=db,
                    container=container,
                )

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid credentials"

    @pytest.mark.asyncio
    async def test_service_key_checked_before_jwt(self):
        """Service key authenticator is tried before JWT authenticator."""
        service_result = AuthResult(user_id=None, auth_type="service")
        container = _make_container()
        db = _make_db()
        jwt_mock = AsyncMock(return_value=AuthResult(user_id="some-id", auth_type="jwt"))

        with (
            patch("baize.auth.deps.authenticate_service_key", return_value=service_result),
            patch("baize.auth.deps.authenticate_jwt", new=jwt_mock),
        ):
            result = await get_auth_result(
                authorization="Bearer token",
                db=db,
                container=container,
            )

        assert result.auth_type == "service"
        jwt_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_jwt_checked_before_api_key(self):
        """JWT authenticator is tried before API key authenticator."""
        user_id = str(uuid.uuid4())
        jwt_result = AuthResult(user_id=user_id, auth_type="jwt")
        container = _make_container()
        db = _make_db()
        api_key_mock = AsyncMock(return_value=AuthResult(user_id="other-id", auth_type="api_key"))

        with (
            patch("baize.auth.deps.authenticate_service_key", return_value=None),
            patch("baize.auth.deps.authenticate_jwt", new=AsyncMock(return_value=jwt_result)),
            patch("baize.auth.deps.authenticate_api_key", new=api_key_mock),
        ):
            result = await get_auth_result(
                authorization="Bearer token",
                db=db,
                container=container,
            )

        assert result.auth_type == "jwt"
        api_key_mock.assert_not_called()


# ---------------------------------------------------------------------------
# get_current_user tests
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    """Tests for the get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_service_auth_raises_403_user_context_required(self):
        """Service key auth raises HTTP 403 with detail='User context required'."""
        auth = AuthResult(user_id=None, auth_type="service")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(auth=auth)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "User context required"

    @pytest.mark.asyncio
    async def test_jwt_auth_returns_user_id(self):
        """Valid JWT auth returns the user_id string."""
        user_id = str(uuid.uuid4())
        auth = AuthResult(user_id=user_id, auth_type="jwt")

        result = await get_current_user(auth=auth)

        assert result == user_id

    @pytest.mark.asyncio
    async def test_api_key_auth_returns_user_id(self):
        """Valid API key auth returns the user_id string."""
        user_id = str(uuid.uuid4())
        auth = AuthResult(user_id=user_id, auth_type="api_key")

        result = await get_current_user(auth=auth)

        assert result == user_id
