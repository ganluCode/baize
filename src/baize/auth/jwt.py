"""JWT authenticator.

Authenticates requests by verifying a JWT's signature and expiry (via
python-jose), then confirming the token is still active in Redis (not revoked).
"""

import logging

import redis.asyncio as aioredis
from jose import JWTError, jwt

from baize.auth.schemas import AuthResult

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"


async def authenticate_jwt(
    authorization: str | None,
    secret_key: str,
    redis: aioredis.Redis,
) -> AuthResult | None:
    """Authenticate a request using a Bearer JWT.

    Verifies the token's signature and expiry, then checks Redis to confirm
    it has not been actively revoked.

    Args:
        authorization: The raw value of the Authorization header, or None.
        secret_key: The HMAC secret used to sign/verify the JWT.
        redis: Async Redis client for revocation checks.

    Returns:
        :class:`AuthResult` with ``auth_type='jwt'`` and the ``user_id`` from
        the token payload if authentication succeeds, otherwise ``None``.
        Never raises an exception.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization[len("Bearer "):]

    try:
        payload = jwt.decode(token, secret_key, algorithms=[_ALGORITHM])
    except JWTError:
        return None

    user_id = payload.get("user_id")
    jti = payload.get("jti")

    if not user_id or not jti:
        return None

    redis_key = f"jwt:{user_id}:{jti}"
    if not await redis.exists(redis_key):
        logger.debug("JWT jti=%s for user_id=%s not found in Redis (revoked or never issued)", jti, user_id)
        return None

    logger.debug("JWT authentication successful for user_id=%s", user_id)
    return AuthResult(user_id=user_id, auth_type="jwt")
