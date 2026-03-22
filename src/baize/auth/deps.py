"""Authentication orchestration FastAPI dependencies.

Provides :func:`get_auth_result` which tries Service Key → JWT → API Key in
order, and :func:`get_current_user` which additionally rejects service-level
authentication (i.e. requires a real user context).
"""

import logging

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from baize.auth.api_key import authenticate_api_key
from baize.auth.jwt import authenticate_jwt
from baize.auth.schemas import AuthResult
from baize.auth.service_key import authenticate_service_key
from baize.core.container import Container
from baize.core.database import get_db
from baize.core.deps import get_container

logger = logging.getLogger(__name__)


async def get_auth_result(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
    container: Container = Depends(get_container),
) -> AuthResult:
    """Orchestrate authentication by trying three methods in priority order.

    Tries Service Key → JWT → API Key.  Returns the first successful
    :class:`~baize.auth.schemas.AuthResult`.

    Args:
        authorization: Raw value of the ``Authorization`` request header.
        db: Async database session (used by the API key authenticator).
        container: Application service container (supplies Redis and config).

    Returns:
        :class:`~baize.auth.schemas.AuthResult` from the first successful
        authenticator.

    Raises:
        HTTPException: 401 with ``detail="Missing credentials"`` when the
            Authorization header is absent.
        HTTPException: 401 with ``detail="Invalid credentials"`` when all
            three authenticators fail.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing credentials")

    # 1. Service Key
    result = authenticate_service_key(authorization)
    if result is not None:
        return result

    # 2. JWT
    result = await authenticate_jwt(
        authorization,
        secret_key=container.config.secret_key,
        redis=container.redis,
    )
    if result is not None:
        return result

    # 3. API Key
    result = await authenticate_api_key(authorization, db)
    if result is not None:
        return result

    raise HTTPException(status_code=401, detail="Invalid credentials")


async def get_current_user(
    auth: AuthResult = Depends(get_auth_result),
) -> str:
    """Resolve the current user ID, rejecting service-level authentication.

    Args:
        auth: Authentication result from :func:`get_auth_result`.

    Returns:
        The authenticated user's ID string.

    Raises:
        HTTPException: 403 with ``detail="User context required"`` when the
            request was authenticated with a Service Key (no user context).
    """
    if auth.auth_type == "service":
        raise HTTPException(status_code=403, detail="User context required")
    return auth.user_id  # type: ignore[return-value]
