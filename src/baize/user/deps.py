"""FastAPI authentication dependency functions for Baize.

Provides reusable dependency functions for JWT, API key, and service key
authentication that can be used with FastAPI's Depends() mechanism.
"""

import hashlib
import logging
import os
import uuid

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from baize.core.container import Container
from baize.core.database import get_db
from baize.core.deps import get_container
from baize.user.jwt_service import TokenError
from baize.user.models import UserModel
from baize.user.repository import UserRepository

logger = logging.getLogger(__name__)


async def get_current_user_jwt(
    authorization: str | None = Header(default=None, alias="Authorization"),
    session: AsyncSession = Depends(get_db),
    container: Container = Depends(get_container),
) -> UserModel:
    """Authenticate a request using a Bearer JWT from the Authorization header.

    Args:
        authorization: The value of the Authorization header.
        session: Async database session.
        container: Application service container (provides JWTService).

    Returns:
        The authenticated UserModel.

    Raises:
        HTTPException: 401 if the header is missing, malformed, the token is
                       invalid/expired/revoked, or the user is not found.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization[7:]

    try:
        payload = await container.jwt_service.verify_token(token)
    except TokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = uuid.UUID(payload["user_id"])
    repo = UserRepository(session)
    user = await repo.get_by_id(user_id)

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def get_current_user_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_db),
) -> UserModel:
    """Authenticate a request using an API key from the X-API-Key header.

    The incoming key is hashed with SHA-256 and compared against the stored
    ``api_key_hash`` in the database.

    Args:
        x_api_key: The value of the X-API-Key header.
        session: Async database session.

    Returns:
        The authenticated UserModel.

    Raises:
        HTTPException: 401 if the header is missing or the key is invalid.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    lookup_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    repo = UserRepository(session)
    user = await repo.get_by_api_key_hash(lookup_hash)

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user


async def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_db),
    container: Container = Depends(get_container),
) -> UserModel:
    """Authenticate a request using JWT (preferred) or API key as fallback.

    Tries the Authorization Bearer token first. If absent or invalid, falls
    back to the X-API-Key header. Raises 401 if both fail.

    Args:
        authorization: The value of the Authorization header.
        x_api_key: The value of the X-API-Key header.
        session: Async database session.
        container: Application service container (provides JWTService).

    Returns:
        The authenticated UserModel.

    Raises:
        HTTPException: 401 if neither authentication method succeeds.
    """
    repo = UserRepository(session)
    bearer_token: str | None = None

    if authorization and authorization.startswith("Bearer "):
        bearer_token = authorization[7:]

    # 1. Try JWT (Bearer token that looks like a JWT)
    if bearer_token:
        try:
            payload = await container.jwt_service.verify_token(bearer_token)
            user_id = uuid.UUID(payload["user_id"])
            user = await repo.get_by_id(user_id)
            if user is not None:
                return user
        except (TokenError, Exception):
            pass

    # 2. Try X-API-Key header
    api_key = x_api_key
    # 3. If no X-API-Key, try Bearer token as API Key (for OpenAI-compatible clients)
    if not api_key and bearer_token:
        api_key = bearer_token

    if api_key:
        lookup_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user = await repo.get_by_api_key_hash(lookup_hash)
        if user is not None:
            return user

    raise HTTPException(status_code=401, detail="Not authenticated")


async def require_admin(user: UserModel = Depends(get_current_user)) -> UserModel:
    """Role guard that restricts access to admin users only.

    Args:
        user: The currently authenticated user.

    Returns:
        The authenticated UserModel (if admin).

    Raises:
        HTTPException: 403 if the user does not have the "admin" role.
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def verify_service_key(
    service_key: str | None = Header(default=None, alias="X-Service-Key"),
) -> None:
    """Verify the service-to-service authentication key from the X-Service-Key header.

    The expected key is read from the ``SERVICE_KEY`` environment variable.
    This is intended for internal service calls that do not have a user context.

    Args:
        service_key: The value of the X-Service-Key header.

    Raises:
        HTTPException: 403 if the key is missing, the env var is not set, or
                       the keys do not match.
    """
    expected = os.environ.get("SERVICE_KEY")
    if not expected or not service_key or service_key != expected:
        raise HTTPException(status_code=403, detail="Invalid service key")
