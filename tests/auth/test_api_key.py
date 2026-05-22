"""Unit tests for API Key authenticator (F-006)."""

import hashlib
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.auth.api_key import authenticate_api_key  # noqa: E402
from baize.auth.schemas import AuthResult  # noqa: E402


def _make_db_session(user_model=None):
    """Build a mock AsyncSession that returns user_model from execute."""
    session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = user_model
    result = AsyncMock()
    result.scalar_one_or_none = scalar_result.scalar_one_or_none
    session.execute = AsyncMock(return_value=result)
    return session


def _make_user(user_id: str, is_active: bool = True):
    """Build a minimal mock UserModel."""
    user = MagicMock()
    user.id = uuid.UUID(user_id)
    user.is_active = is_active
    return user


@pytest.mark.asyncio
async def test_valid_api_key_returns_auth_result():
    """Valid active API key returns AuthResult with correct user_id."""
    user_id = str(uuid.uuid4())
    plaintext_key = "valid-api-key-12345"
    user = _make_user(user_id, is_active=True)
    session = _make_db_session(user_model=user)

    result = await authenticate_api_key(f"Bearer {plaintext_key}", session)

    assert result is not None
    assert isinstance(result, AuthResult)
    assert result.user_id == user_id
    assert result.auth_type == "api_key"


@pytest.mark.asyncio
async def test_nonexistent_api_key_returns_none():
    """API key whose hash is not in the database returns None."""
    plaintext_key = "nonexistent-key"
    session = _make_db_session(user_model=None)

    result = await authenticate_api_key(f"Bearer {plaintext_key}", session)

    assert result is None


@pytest.mark.asyncio
async def test_disabled_user_api_key_returns_none():
    """API key belonging to an inactive user returns None."""
    user_id = str(uuid.uuid4())
    plaintext_key = "disabled-user-key"
    user = _make_user(user_id, is_active=False)
    session = _make_db_session(user_model=user)

    result = await authenticate_api_key(f"Bearer {plaintext_key}", session)

    assert result is None


@pytest.mark.asyncio
async def test_missing_bearer_prefix_returns_none():
    """Authorization header without 'Bearer ' prefix returns None."""
    session = _make_db_session(user_model=None)

    result = await authenticate_api_key("plain-token", session)

    assert result is None


@pytest.mark.asyncio
async def test_none_authorization_returns_none():
    """None authorization header returns None."""
    session = _make_db_session(user_model=None)

    result = await authenticate_api_key(None, session)

    assert result is None


@pytest.mark.asyncio
async def test_hash_used_for_db_lookup_is_sha256():
    """The database lookup uses sha256(plaintext_key) hex digest."""
    plaintext_key = "test-key-for-hash-check"
    expected_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

    user_id = str(uuid.uuid4())
    user = _make_user(user_id, is_active=True)
    session = _make_db_session(user_model=user)

    await authenticate_api_key(f"Bearer {plaintext_key}", session)

    # The session.execute must have been called once
    session.execute.assert_called_once()
    call_args = session.execute.call_args[0][0]
    # The compiled WHERE clause should contain the expected hash
    compiled = str(call_args.compile(compile_kwargs={"literal_binds": True}))
    assert expected_hash in compiled


@pytest.mark.asyncio
async def test_user_id_matches_database_record():
    """Returned user_id matches the UUID from the database record."""
    user_id = str(uuid.uuid4())
    plaintext_key = "another-valid-key"
    user = _make_user(user_id, is_active=True)
    session = _make_db_session(user_model=user)

    result = await authenticate_api_key(f"Bearer {plaintext_key}", session)

    assert result is not None
    assert result.user_id == user_id
