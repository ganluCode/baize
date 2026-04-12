"""JWT service for issuing, verifying, and revoking access tokens.

Tokens are backed by Redis for active revocation support.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

import redis.asyncio as aioredis
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"
_TOKEN_TTL_SECONDS = 86400  # 24 hours


class TokenError(Exception):
    """Raised when a JWT is invalid, expired, or has been revoked."""


class JWTService:
    """Issues and validates JWTs backed by Redis for revocation support."""

    def __init__(self, secret_key: str, redis: aioredis.Redis) -> None:
        self._secret = secret_key
        self._redis = redis

    async def create_token(self, user_id: uuid.UUID, role: str) -> str:
        """Issue a signed JWT and persist it in Redis.

        Args:
            user_id: The user's UUID.
            role: The user's role (e.g. "user" or "admin").

        Returns:
            Signed JWT string.
        """
        jti = str(uuid.uuid4())
        now = datetime.now(tz=UTC)
        exp = now + timedelta(seconds=_TOKEN_TTL_SECONDS)

        payload = {
            "user_id": str(user_id),
            "role": role,
            "exp": exp,
            "jti": jti,
        }

        token = jwt.encode(payload, self._secret, algorithm=_ALGORITHM)
        redis_key = f"jwt:{user_id}:{jti}"
        await self._redis.setex(redis_key, _TOKEN_TTL_SECONDS, "1")
        logger.debug("Issued JWT jti=%s for user_id=%s", jti, user_id)
        return token

    async def verify_token(self, token: str) -> dict:
        """Verify a JWT's signature and Redis presence.

        Args:
            token: The JWT string to verify.

        Returns:
            Decoded payload dict containing user_id, role, exp, jti.

        Raises:
            TokenError: If the signature is invalid, token is expired,
                        or the token has been revoked (not in Redis).
        """
        try:
            payload = jwt.decode(token, self._secret, algorithms=[_ALGORITHM])
        except JWTError as exc:
            raise TokenError(f"Invalid token: {exc}") from exc

        user_id = payload.get("user_id")
        jti = payload.get("jti")
        redis_key = f"jwt:{user_id}:{jti}"

        if not await self._redis.exists(redis_key):
            raise TokenError("Token has been revoked or does not exist in store.")

        return payload

    async def revoke_token(self, user_id: uuid.UUID, jti: str) -> None:
        """Remove a JWT from Redis, rendering it invalid for future requests.

        Args:
            user_id: The user's UUID (used to construct the Redis key).
            jti: The JWT ID claim value.
        """
        redis_key = f"jwt:{user_id}:{jti}"
        await self._redis.delete(redis_key)
        logger.debug("Revoked JWT jti=%s for user_id=%s", jti, user_id)
