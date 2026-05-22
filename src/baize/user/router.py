"""User router: user profile and admin user management endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.repository import AgentConfigRepository
from baize.agent.service import AgentConfigService
from baize.core.database import get_db
from baize.session.repository import ChatMessageRepository, SessionRepository
from baize.session.service import SessionService
from baize.user.deps import get_current_user, require_admin
from baize.user.models import UserModel
from baize.user.repository import UserRepository
from baize.user.schemas import (
    ResetKeyResponse,
    UserCreateRequest,
    UserCreateResponse,
    UserListResponse,
    UserResponse,
    UserUpdateRequest,
)
from baize.user.service import UserService, UserServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


async def _get_user_service(session: AsyncSession = Depends(get_db)) -> UserService:
    """Build a per-request UserService from a DB session."""
    repo = UserRepository(session)
    agent_repo = AgentConfigRepository(session)
    session_repo = SessionRepository(session)
    message_repo = ChatMessageRepository(session)
    session_svc = SessionService(session_repo=session_repo, message_repo=message_repo)
    agent_svc = AgentConfigService(repository=agent_repo, session_service=session_svc)
    return UserService(user_repo=repo, agent_config_service=agent_svc)


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: UserModel = Depends(get_current_user),
    svc: UserService = Depends(_get_user_service),
) -> UserResponse:
    """Return the current user's profile.

    Returns:
        UserResponse without password or api_key_hash.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    return await svc.get_me(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UserUpdateRequest,
    current_user: UserModel = Depends(get_current_user),
    svc: UserService = Depends(_get_user_service),
) -> UserResponse:
    """Update the current user's profile.

    Returns:
        Updated UserResponse.

    Raises:
        HTTPException: 401 if not authenticated; 409 if name is taken.
    """
    try:
        return await svc.update_me(current_user, body)
    except UserServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


@router.post("", response_model=UserCreateResponse, status_code=201)
async def create_user(
    body: UserCreateRequest,
    _admin: UserModel = Depends(require_admin),
    svc: UserService = Depends(_get_user_service),
) -> UserCreateResponse:
    """Create a new user (admin only).

    Returns:
        UserCreateResponse with one-time plaintext api_key.

    Raises:
        HTTPException: 401 if not authenticated; 403 if not admin.
    """
    return await svc.create_user(body)


@router.get("", response_model=UserListResponse)
async def list_users(
    skip: int = 0,
    limit: int = 20,
    _admin: UserModel = Depends(require_admin),
    svc: UserService = Depends(_get_user_service),
) -> UserListResponse:
    """List all users with pagination (admin only).

    Returns:
        UserListResponse with items and total count.

    Raises:
        HTTPException: 401 if not authenticated; 403 if not admin.
    """
    return await svc.list_users(skip=skip, limit=limit)


@router.post("/{user_id}/reset-key", response_model=ResetKeyResponse)
async def reset_api_key(
    user_id: uuid.UUID,
    _admin: UserModel = Depends(require_admin),
    svc: UserService = Depends(_get_user_service),
) -> ResetKeyResponse:
    """Reset the API key for a user (admin only).

    Returns:
        ResetKeyResponse with new plaintext API key.

    Raises:
        HTTPException: 401 if not authenticated; 403 if not admin; 404 if user not found.
    """
    try:
        return await svc.reset_api_key(user_id)
    except UserServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
