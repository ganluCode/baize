"""Tests for FastAPI authentication dependencies (src/baize/user/deps.py)."""

import hashlib
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from fastapi import HTTPException  # noqa: E402

from baize.user.deps import (  # noqa: E402
    get_current_user,
    get_current_user_api_key,
    get_current_user_jwt,
    require_admin,
    verify_service_key,
)
from baize.user.jwt_service import TokenError  # noqa: E402
from baize.user.models import UserModel  # noqa: E402


def _make_user(role: str = "user") -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = uuid.uuid4()
    user.role = role
    return user


def _make_container(jwt_verify_return=None, jwt_side_effect=None):
    container = MagicMock()
    mock_jwt = AsyncMock()
    if jwt_side_effect:
        mock_jwt.verify_token.side_effect = jwt_side_effect
    else:
        mock_jwt.verify_token.return_value = jwt_verify_return
    container.jwt_service = mock_jwt
    return container


def _make_session(user_return=None):
    session = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# get_current_user_jwt
# ---------------------------------------------------------------------------


async def test_get_current_user_jwt_valid_token_returns_user():
    user = _make_user()
    payload = {"user_id": str(user.id), "role": "user", "jti": "jti123"}
    container = _make_container(jwt_verify_return=payload)

    mock_repo = AsyncMock()
    mock_repo.get_by_id.return_value = user
    session = _make_session()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        result = await get_current_user_jwt(
            authorization="Bearer validtoken",
            session=session,
            container=container,
        )

    assert result is user
    mock_repo.get_by_id.assert_awaited_once_with(user.id)


async def test_get_current_user_jwt_missing_header_returns_401():
    container = _make_container(jwt_verify_return={})
    session = _make_session()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_jwt(authorization=None, session=session, container=container)

    assert exc_info.value.status_code == 401


async def test_get_current_user_jwt_malformed_header_returns_401():
    container = _make_container(jwt_verify_return={})
    session = _make_session()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_jwt(
            authorization="Token notbearer",
            session=session,
            container=container,
        )

    assert exc_info.value.status_code == 401


async def test_get_current_user_jwt_invalid_token_returns_401():
    container = _make_container(jwt_side_effect=TokenError("bad token"))
    session = _make_session()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_jwt(
            authorization="Bearer badtoken",
            session=session,
            container=container,
        )

    assert exc_info.value.status_code == 401


async def test_get_current_user_jwt_user_not_found_returns_401():
    user_id = uuid.uuid4()
    payload = {"user_id": str(user_id), "role": "user", "jti": "jti123"}
    container = _make_container(jwt_verify_return=payload)

    mock_repo = AsyncMock()
    mock_repo.get_by_id.return_value = None
    session = _make_session()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_jwt(
                authorization="Bearer sometoken",
                session=session,
                container=container,
            )

    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# get_current_user_api_key
# ---------------------------------------------------------------------------


async def test_get_current_user_api_key_valid_key_returns_user():
    user = _make_user()
    api_key = "test-api-key-12345"
    expected_hash = hashlib.sha256(api_key.encode()).hexdigest()

    mock_repo = AsyncMock()
    mock_repo.get_by_api_key_hash.return_value = user
    session = _make_session()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        result = await get_current_user_api_key(x_api_key=api_key, session=session)

    assert result is user
    mock_repo.get_by_api_key_hash.assert_awaited_once_with(expected_hash)


async def test_get_current_user_api_key_missing_header_returns_401():
    session = _make_session()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_api_key(x_api_key=None, session=session)

    assert exc_info.value.status_code == 401


async def test_get_current_user_api_key_invalid_key_returns_401():
    mock_repo = AsyncMock()
    mock_repo.get_by_api_key_hash.return_value = None
    session = _make_session()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_api_key(x_api_key="wrong-key", session=session)

    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# get_current_user (tries JWT first, then API key)
# ---------------------------------------------------------------------------


async def test_get_current_user_with_valid_jwt_returns_user():
    user = _make_user()
    payload = {"user_id": str(user.id), "role": "user", "jti": "jti123"}
    container = _make_container(jwt_verify_return=payload)

    mock_repo = AsyncMock()
    mock_repo.get_by_id.return_value = user
    session = _make_session()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        result = await get_current_user(
            authorization="Bearer validtoken",
            x_api_key=None,
            session=session,
            container=container,
        )

    assert result is user


async def test_get_current_user_falls_back_to_api_key_when_no_jwt():
    user = _make_user()
    api_key = "my-api-key"
    expected_hash = hashlib.sha256(api_key.encode()).hexdigest()

    mock_repo = AsyncMock()
    mock_repo.get_by_api_key_hash.return_value = user
    session = _make_session()

    # No Authorization header, container not needed
    container = MagicMock()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        result = await get_current_user(
            authorization=None,
            x_api_key=api_key,
            session=session,
            container=container,
        )

    assert result is user
    mock_repo.get_by_api_key_hash.assert_awaited_once_with(expected_hash)


async def test_get_current_user_falls_back_to_api_key_when_jwt_invalid():
    user = _make_user()
    api_key = "fallback-api-key"

    container = _make_container(jwt_side_effect=TokenError("expired"))
    mock_repo = AsyncMock()
    mock_repo.get_by_api_key_hash.return_value = user
    session = _make_session()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        result = await get_current_user(
            authorization="Bearer expiredtoken",
            x_api_key=api_key,
            session=session,
            container=container,
        )

    assert result is user


async def test_get_current_user_both_invalid_returns_401():
    container = _make_container(jwt_side_effect=TokenError("bad"))
    mock_repo = AsyncMock()
    mock_repo.get_by_api_key_hash.return_value = None
    session = _make_session()

    with patch("baize.user.deps.UserRepository", return_value=mock_repo):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                authorization="Bearer bad",
                x_api_key="badkey",
                session=session,
                container=container,
            )

    assert exc_info.value.status_code == 401


async def test_get_current_user_no_credentials_returns_401():
    container = MagicMock()
    session = _make_session()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(
            authorization=None,
            x_api_key=None,
            session=session,
            container=container,
        )

    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# require_admin
# ---------------------------------------------------------------------------


async def test_require_admin_with_admin_role_returns_user():
    user = _make_user(role="admin")
    result = await require_admin(user=user)
    assert result is user


async def test_require_admin_with_non_admin_role_raises_403():
    user = _make_user(role="user")

    with pytest.raises(HTTPException) as exc_info:
        await require_admin(user=user)

    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# verify_service_key
# ---------------------------------------------------------------------------


async def test_verify_service_key_matching_key_does_not_raise():
    with patch.dict(os.environ, {"SERVICE_KEY": "supersecret"}):
        # Should not raise
        await verify_service_key(service_key="supersecret")


async def test_verify_service_key_mismatched_key_raises_403():
    with patch.dict(os.environ, {"SERVICE_KEY": "supersecret"}):
        with pytest.raises(HTTPException) as exc_info:
            await verify_service_key(service_key="wrongkey")

    assert exc_info.value.status_code == 403


async def test_verify_service_key_missing_env_var_raises_403():
    env_without_key = {k: v for k, v in os.environ.items() if k != "SERVICE_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await verify_service_key(service_key="anykey")

    assert exc_info.value.status_code == 403
