"""Knowledge Base CRUD API routes."""

import hashlib
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi import status as http_status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.core.database import get_db
from baize.core.deps import get_provider_factory
from baize.knowledge.deps import get_knowledge_base_service
from baize.knowledge.models import KnowledgeDocumentModel
from baize.knowledge.schemas import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentResponse,
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


@router.post(
    "/{kb_id}/documents",
    response_model=KnowledgeDocumentResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
)
async def create_document_json(
    kb_id: uuid.UUID,
    body: KnowledgeDocumentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """Create a document in a knowledge base via JSON body.

    The document is created synchronously with status 'pending' and ingestion
    is queued as a background task that processes parse/chunk/embed/store.

    Returns:
        202 Accepted + KnowledgeDocumentResponse with status='pending'.

    Raises:
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found or deleted.
        HTTPException: 409 if a document with the same content already exists (ingested).
        HTTPException: 422 if request body is invalid.
    """
    logger.info(
        "POST /knowledge-bases/%s/documents user_id=%s", kb_id, current_user.id
    )

    # Validate KB exists and belongs to current user
    try:
        await svc.get(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc

    # Compute content hash for deduplication
    content_hash = hashlib.sha256(body.content.encode("utf-8")).hexdigest()

    # Check if same content is already ingested in this KB
    dup_result = await db.execute(
        select(KnowledgeDocumentModel).where(
            KnowledgeDocumentModel.kb_id == kb_id,
            KnowledgeDocumentModel.content_hash == content_hash,
            KnowledgeDocumentModel.status == "ingested",
        )
    )
    existing = dup_result.scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={
                "message": "Document with the same content already ingested in this knowledge base.",
                "existing_doc_id": str(existing.id),
            },
        )

    # Create document record with status=pending
    now = datetime.now(UTC)
    doc = KnowledgeDocumentModel(
        id=uuid.uuid4(),
        kb_id=kb_id,
        title=body.title,
        source_type=body.source_type,
        source_uri=body.source_uri,
        content_hash=content_hash,
        doc_metadata=body.doc_metadata if body.doc_metadata is not None else {},
        status="pending",
        chunk_count=0,
        created_at=now,
        updated_at=now,
    )
    db.add(doc)
    await db.commit()

    # Queue ingestion as background task
    background_tasks.add_task(
        _bg_ingest_document,
        doc_id=doc.id,
        kb_id=kb_id,
        content=body.content,
    )

    return KnowledgeDocumentResponse.model_validate(doc)


async def _bg_ingest_document(
    *,
    doc_id: uuid.UUID,
    kb_id: uuid.UUID,
    content: str,
) -> None:
    """Background task: process a pending document through the ingestion pipeline."""
    from baize.core.database import AsyncSessionLocal
    from baize.core.deps import get_container
    from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
    from baize.knowledge.ingestion.qdrant_client import get_qdrant_client
    from baize.knowledge.ingestion.service import IngestionService
    from baize.knowledge.models import KnowledgeBaseModel

    async with AsyncSessionLocal() as session:
        kb = await session.get(KnowledgeBaseModel, kb_id)
        if kb is None:
            logger.error(
                "Background ingestion: KB %s not found for doc %s", kb_id, doc_id
            )
            return

        container = get_container()
        embedder = KnowledgeEmbedder(
            provider_name=kb.embedding_provider,
            model_id=kb.embedding_model,
            expected_dim=kb.embedding_dim,
            llm_provider_manager=container.provider_factory,
        )
        qdrant = await get_qdrant_client()
        ingestion_svc = IngestionService(session=session, embedder=embedder, qdrant=qdrant)

        try:
            await ingestion_svc.process_pending_document(
                doc_id=doc_id,
                kb_id=kb_id,
                content=content,
            )
        except Exception:
            logger.exception("Background ingestion failed for document %s", doc_id)
