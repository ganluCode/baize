"""Unit tests for JWT authenticator (F-005)."""

import os
import uuid
from datetime import datetime, timedelta, timezone

import fakeredis.aioredis
import pytest
from jose import jwt

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-jwt-tests")

from baize.auth.jwt import authenticate_jwt  # noqa: E402
from baize.auth.schemas import AuthResult  # noqa: E402

_SECRET = "test-secret-for-jwt-auth"
_ALGORITHM = "HS256"


def _make_token(user_id: str, jti: str, expires_delta: timedelta = timedelta(hours=1)) -> str:
    """Helper to create a signed JWT."""
    now = datetime.now(tz=timezone.utc)
    payload = {
        "user_id": user_id,
        "role": "user",
        "exp": now + expires_delta,
        "jti": jti,
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.mark.asyncio
async def test_valid_token_with_redis_key_returns_auth_result(fake_redis):
    """Valid JWT with matching Redis key returns AuthResult with correct user_id."""
    user_id = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    token = _make_token(user_id, jti)
    redis_key = f"jwt:{user_id}:{jti}"
    await fake_redis.setex(redis_key, 3600, "1")

    result = await authenticate_jwt(f"Bearer {token}", _SECRET, fake_redis)

    assert result is not None
    assert isinstance(result, AuthResult)
    assert result.user_id == user_id
    assert result.auth_type == "jwt"


@pytest.mark.asyncio
async def test_invalid_signature_returns_none(fake_redis):
    """JWT with tampered signature returns None without raising."""
    user_id = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    token = _make_token(user_id, jti)
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + ".invalidsignature"

    result = await authenticate_jwt(f"Bearer {tampered}", _SECRET, fake_redis)

    assert result is None


@pytest.mark.asyncio
async def test_expired_token_returns_none(fake_redis):
    """Expired JWT returns None."""
    user_id = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    token = _make_token(user_id, jti, expires_delta=timedelta(seconds=-1))

    result = await authenticate_jwt(f"Bearer {token}", _SECRET, fake_redis)

    assert result is None


@pytest.mark.asyncio
async def test_revoked_token_not_in_redis_returns_none(fake_redis):
    """JWT whose Redis key has been deleted (revoked) returns None."""
    user_id = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    token = _make_token(user_id, jti)
    # Do NOT write the key to Redis — simulates a revoked/never-issued token

    result = await authenticate_jwt(f"Bearer {token}", _SECRET, fake_redis)

    assert result is None


@pytest.mark.asyncio
async def test_non_bearer_authorization_returns_none(fake_redis):
    """Authorization header without 'Bearer ' prefix returns None."""
    user_id = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    token = _make_token(user_id, jti)

    result = await authenticate_jwt(token, _SECRET, fake_redis)

    assert result is None


@pytest.mark.asyncio
async def test_none_authorization_returns_none(fake_redis):
    """None authorization header returns None."""
    result = await authenticate_jwt(None, _SECRET, fake_redis)
    assert result is None


@pytest.mark.asyncio
async def test_user_id_matches_token_payload(fake_redis):
    """Returned user_id matches the user_id field in the JWT payload."""
    user_id = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    token = _make_token(user_id, jti)
    redis_key = f"jwt:{user_id}:{jti}"
    await fake_redis.setex(redis_key, 3600, "1")

    result = await authenticate_jwt(f"Bearer {token}", _SECRET, fake_redis)

    assert result is not None
    assert result.user_id == user_id
