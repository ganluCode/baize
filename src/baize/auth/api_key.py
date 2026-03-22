"""API Key authenticator.

Authenticates requests by computing sha256(token) and querying the
system_users table for a matching api_key_hash. Requires an active user.
"""

import hashlib
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.auth.schemas import AuthResult
from baize.user.models import UserModel

logger = logging.getLogger(__name__)


async def authenticate_api_key(
    authorization: str | None,
    db: AsyncSession,
) -> AuthResult | None:
    """Authenticate a request using a Bearer API key.

    Extracts the token from the ``Authorization: Bearer <token>`` header,
    computes its sha256 hex digest, and queries the database for a matching
    active user.

    Args:
        authorization: The raw value of the Authorization header, or None.
        db: Async SQLAlchemy session for database access.

    Returns:
        :class:`AuthResult` with ``auth_type='api_key'`` and the user's ID
        if authentication succeeds, otherwise ``None``.  Never raises.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization[len("Bearer "):]
    key_hash = hashlib.sha256(token.encode()).hexdigest()

    result = await db.execute(
        select(UserModel).where(UserModel.api_key_hash == key_hash)
    )
    user = result.scalar_one_or_none()

    if user is None:
        return None

    if not user.is_active:
        logger.debug("API key authentication rejected: user %s is inactive.", user.id)
        return None

    logger.debug("API key authentication successful for user_id=%s", user.id)
    return AuthResult(user_id=str(user.id), auth_type="api_key")
