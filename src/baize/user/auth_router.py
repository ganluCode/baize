"""Auth router: POST /api/v1/auth/login, /logout, /refresh."""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from baize.core.container import Container
from baize.core.database import get_db
from baize.core.deps import get_container
from baize.user.auth_service import AuthError, AuthService
from baize.user.repository import UserRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Request body for the login endpoint."""

    login: str
    password: str


class TokenResponse(BaseModel):
    """Token response returned by login and refresh."""

    access_token: str
    token_type: str
    expires_in: int


def _get_bearer_token(authorization: str | None = Header(default=None, alias="Authorization")) -> str:
    """Extract the bearer token from the Authorization header.

    Raises:
        HTTPException: 401 if the header is missing or malformed.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return authorization[7:]


async def _get_auth_service(
    session: AsyncSession = Depends(get_db),
    container: Container = Depends(get_container),
) -> AuthService:
    """Build a per-request AuthService from the container's JWT service and a DB session."""
    repo = UserRepository(session)
    return AuthService(user_repo=repo, jwt_service=container.jwt_service)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    auth_service: AuthService = Depends(_get_auth_service),
) -> TokenResponse:
    """Authenticate a user and return a JWT access token.

    Returns:
        TokenResponse with access_token, token_type, and expires_in.

    Raises:
        HTTPException: 401 if credentials are invalid; 403 if account is inactive.
    """
    try:
        result = await auth_service.login(login=body.login, password=body.password)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return TokenResponse(
        access_token=result.access_token,
        token_type=result.token_type,
        expires_in=result.expires_in,
    )


@router.post("/logout")
async def logout(
    token: str = Depends(_get_bearer_token),
    auth_service: AuthService = Depends(_get_auth_service),
) -> dict[str, str]:
    """Revoke the current JWT, preventing further use.

    Returns:
        Confirmation message on success.

    Raises:
        HTTPException: 401 if the token is missing, malformed, or already invalid.
    """
    try:
        await auth_service.logout(token=token)
    except Exception as exc:
        logger.debug("Logout failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"detail": "Successfully logged out."}


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    token: str = Depends(_get_bearer_token),
    auth_service: AuthService = Depends(_get_auth_service),
) -> TokenResponse:
    """Issue a new JWT and revoke the old one.

    Returns:
        TokenResponse containing the new access token.

    Raises:
        HTTPException: 401 if the token is missing, malformed, or already invalid.
    """
    try:
        result = await auth_service.refresh(token=token)
    except Exception as exc:
        logger.debug("Refresh failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return TokenResponse(
        access_token=result.access_token,
        token_type=result.token_type,
        expires_in=result.expires_in,
    )
