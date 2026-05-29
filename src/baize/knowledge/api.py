"""Knowledge Base CRUD API routes."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status as http_status

from baize.core.deps import get_provider_factory
from baize.knowledge.deps import get_knowledge_base_service
from baize.knowledge.schemas import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
)
from baize.knowledge.service import KnowledgeBaseService, KnowledgeBaseServiceError
from baize.llm.provider import ProviderFactory
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge"])


def _service_error_to_http(exc: KnowledgeBaseServiceError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.post(
    "",
    response_model=KnowledgeBaseResponse,
    status_code=http_status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    body: KnowledgeBaseCreateRequest,
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> KnowledgeBaseResponse:
    """Create a new knowledge base for the authenticated user.

    Returns:
        KnowledgeBaseResponse with status 201.

    Raises:
        HTTPException: 400 if embedding provider/model not found or unavailable.
        HTTPException: 409 if a KB with the same name already exists for this user.
    """
    logger.info("POST /knowledge-bases user_id=%s", current_user.id)
    try:
        return await svc.create(
            user_id=current_user.id, data=body, provider_factory=provider_factory
        )
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc


@router.get("", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    kb_status: str = Query(default="active", alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseListResponse:
    """List knowledge bases for the authenticated user.

    Supports status/limit/offset query parameters; default status is 'active'.

    Returns:
        KnowledgeBaseListResponse with status 200.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    logger.info(
        "GET /knowledge-bases user_id=%s status=%s limit=%d offset=%d",
        current_user.id,
        kb_status,
        limit,
        offset,
    )
    return await svc.list(
        user_id=current_user.id, status=kb_status, limit=limit, offset=offset
    )


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseResponse:
    """Return a single knowledge base owned by the authenticated user.

    Returns:
        KnowledgeBaseResponse with status 200.

    Raises:
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found or status is 'deleted'.
    """
    logger.info("GET /knowledge-bases/%s user_id=%s", kb_id, current_user.id)
    try:
        return await svc.get(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc


@router.delete("/{kb_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> None:
    """Soft-delete a knowledge base by setting its status to 'deleted'.

    Returns:
        204 No Content.

    Raises:
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found or already deleted.
    """
    logger.info("DELETE /knowledge-bases/%s user_id=%s", kb_id, current_user.id)
    try:
        await svc.soft_delete(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc
