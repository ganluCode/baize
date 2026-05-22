"""Unit tests for baize.user.jwt_service using fakeredis."""

import os
import uuid

import fakeredis.aioredis
import pytest
from jose import jwt

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-jwt-unit-tests")

from baize.user.jwt_service import JWTService, TokenError  # noqa: E402


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def jwt_service(fake_redis):
    return JWTService(secret_key="test-secret-for-jwt", redis=fake_redis)


@pytest.mark.asyncio
async def test_create_token_returns_jwt_string(jwt_service):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    assert isinstance(token, str)
    assert len(token) > 0


@pytest.mark.asyncio
async def test_create_token_payload_contains_required_fields(jwt_service):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="admin")
    payload = jwt.decode(token, "test-secret-for-jwt", algorithms=["HS256"])
    assert str(user_id) == payload["user_id"]
    assert payload["role"] == "admin"
    assert "exp" in payload
    assert "jti" in payload


@pytest.mark.asyncio
async def test_create_token_writes_to_redis(jwt_service, fake_redis):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    payload = jwt.decode(token, "test-secret-for-jwt", algorithms=["HS256"])
    key = f"jwt:{user_id}:{payload['jti']}"
    exists = await fake_redis.exists(key)
    assert exists == 1


@pytest.mark.asyncio
async def test_create_token_redis_ttl_within_24h(jwt_service, fake_redis):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    payload = jwt.decode(token, "test-secret-for-jwt", algorithms=["HS256"])
    key = f"jwt:{user_id}:{payload['jti']}"
    ttl = await fake_redis.ttl(key)
    assert 0 < ttl <= 86400


@pytest.mark.asyncio
async def test_verify_token_success(jwt_service):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    result = await jwt_service.verify_token(token)
    assert result["user_id"] == str(user_id)
    assert result["role"] == "user"


@pytest.mark.asyncio
async def test_verify_token_invalid_signature_raises(jwt_service):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    # Tamper with the token signature
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + ".invalidsignature"
    with pytest.raises(TokenError):
        await jwt_service.verify_token(tampered)


@pytest.mark.asyncio
async def test_verify_token_missing_from_redis_raises(jwt_service, fake_redis):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    payload = jwt.decode(token, "test-secret-for-jwt", algorithms=["HS256"])
    # Delete from redis to simulate revocation
    await fake_redis.delete(f"jwt:{user_id}:{payload['jti']}")
    with pytest.raises(TokenError):
        await jwt_service.verify_token(token)


@pytest.mark.asyncio
async def test_revoke_token_then_verify_raises(jwt_service):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    payload = jwt.decode(token, "test-secret-for-jwt", algorithms=["HS256"])
    await jwt_service.revoke_token(user_id=user_id, jti=payload["jti"])
    with pytest.raises(TokenError):
        await jwt_service.verify_token(token)


@pytest.mark.asyncio
async def test_revoke_token_removes_redis_key(jwt_service, fake_redis):
    user_id = uuid.uuid4()
    token = await jwt_service.create_token(user_id=user_id, role="user")
    payload = jwt.decode(token, "test-secret-for-jwt", algorithms=["HS256"])
    key = f"jwt:{user_id}:{payload['jti']}"
    assert await fake_redis.exists(key) == 1
    await jwt_service.revoke_token(user_id=user_id, jti=payload["jti"])
    assert await fake_redis.exists(key) == 0
