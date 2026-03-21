"""Session router: session CRUD API endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends, status

from baize.core.deps import get_session_service
from baize.session.models import SessionStatus
from baize.session.schemas import SessionCreate, SessionListResponse, SessionResponse, SessionUpdate
from baize.session.service import SessionService
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["sessions"])
session_router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "/{agent_id}/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    agent_id: uuid.UUID,
    body: SessionCreate,
    current_user: UserModel = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Create a new session for the authenticated user under the given agent.

    Returns:
        SessionResponse with status 201.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    session = await svc.create_session(
        user_id=current_user.id,
        agent_id=agent_id,
        title=body.title,
    )
    return SessionResponse.model_validate(session)


@router.get("/{agent_id}/sessions", response_model=SessionListResponse)
async def list_sessions(
    agent_id: uuid.UUID,
    status: SessionStatus | None = None,
    limit: int = 20,
    offset: int = 0,
    current_user: UserModel = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
) -> SessionListResponse:
    """Return paginated sessions for the authenticated user under the given agent.

    Sessions are ordered by updated_at descending.

    Returns:
        SessionListResponse with items and total count.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    items, total = await svc.list_sessions(
        user_id=current_user.id,
        agent_id=agent_id,
        status=status.value if status is not None else None,
        limit=limit,
        offset=offset,
    )
    return SessionListResponse(
        items=[SessionResponse.model_validate(s) for s in items],
        total=total,
    )


@session_router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Return a single session owned by the authenticated user.

    Returns:
        SessionResponse with status 200.

    Raises:
        HTTPException: 401 if not authenticated, 404 if not found, 403 if not owner.
    """
    session = await svc.get_session(session_id=session_id, user_id=current_user.id)
    return SessionResponse.model_validate(session)


@session_router.patch("/{session_id}", response_model=SessionResponse)
async def archive_session(
    session_id: uuid.UUID,
    body: SessionUpdate,
    current_user: UserModel = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Archive a session owned by the authenticated user.

    Returns:
        SessionResponse with updated status.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not owner, 422 if invalid status.
    """
    session = await svc.archive_session(session_id=session_id, user_id=current_user.id)
    return SessionResponse.model_validate(session)


@session_router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: SessionService = Depends(get_session_service),
) -> None:
    """Delete a session owned by the authenticated user.

    Returns:
        204 No Content.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not owner.
    """
    await svc.delete_session(session_id=session_id, user_id=current_user.id)
