"""AgentConfig CRUD API routes."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from baize.agent.schemas import AgentCreate, AgentResponse, AgentUpdate
from baize.agent.service import AgentConfigService, AgentServiceError
from baize.core.deps import get_agent_config_service
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


def _service_error_to_http(exc: AgentServiceError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    body: AgentCreate,
    current_user: UserModel = Depends(get_current_user),
    svc: AgentConfigService = Depends(get_agent_config_service),
) -> AgentResponse:
    """Create a new agent configuration for the authenticated user.

    Returns:
        AgentResponse with status 201.

    Raises:
        HTTPException: 401 if not authenticated, 400 if tools are invalid.
    """
    try:
        agent = await svc.create(user_id=current_user.id, data=body)
    except AgentServiceError as exc:
        raise _service_error_to_http(exc) from exc
    return AgentResponse.model_validate(agent)


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    current_user: UserModel = Depends(get_current_user),
    svc: AgentConfigService = Depends(get_agent_config_service),
) -> list[AgentResponse]:
    """Return all agent configurations owned by the authenticated user.

    Returns:
        List of AgentResponse with status 200.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    agents = await svc.list(user_id=current_user.id)
    return [AgentResponse.model_validate(a) for a in agents]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: AgentConfigService = Depends(get_agent_config_service),
) -> AgentResponse:
    """Return a single agent configuration owned by the authenticated user.

    Returns:
        AgentResponse with status 200.

    Raises:
        HTTPException: 401 if not authenticated, 404 if not found or not owned.
    """
    try:
        agent = await svc.get(agent_id=agent_id, user_id=current_user.id)
    except AgentServiceError as exc:
        raise _service_error_to_http(exc) from exc
    return AgentResponse.model_validate(agent)


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: uuid.UUID,
    body: AgentUpdate,
    current_user: UserModel = Depends(get_current_user),
    svc: AgentConfigService = Depends(get_agent_config_service),
) -> AgentResponse:
    """Partially update an agent configuration owned by the authenticated user.

    Returns:
        Updated AgentResponse with status 200.

    Raises:
        HTTPException: 401 if not authenticated, 404 if not found, 400 if tools invalid.
    """
    try:
        agent = await svc.update(agent_id=agent_id, user_id=current_user.id, data=body)
    except AgentServiceError as exc:
        raise _service_error_to_http(exc) from exc
    return AgentResponse.model_validate(agent)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: AgentConfigService = Depends(get_agent_config_service),
) -> None:
    """Delete an agent configuration owned by the authenticated user.

    Returns:
        204 No Content.

    Raises:
        HTTPException: 401 if not authenticated, 404 if not found, 400 if default agent.
    """
    try:
        await svc.delete(agent_id=agent_id, user_id=current_user.id)
    except AgentServiceError as exc:
        raise _service_error_to_http(exc) from exc
