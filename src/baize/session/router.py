"""Session router: create and list session API endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends, status

from baize.core.deps import get_session_service
from baize.session.models import SessionStatus
from baize.session.schemas import SessionCreate, SessionListResponse, SessionResponse
from baize.session.service import SessionService
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["sessions"])


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
