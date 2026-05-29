"""Knowledge Base CRUD API routes."""

import hashlib
import io
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi import status as http_status
from pypdf import PdfReader
from qdrant_client.models import PointIdsList
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.core.database import get_db
from baize.core.deps import get_provider_factory
from baize.knowledge.deps import get_knowledge_base_service, get_retrieval_service
from baize.knowledge.ingestion.qdrant_client import get_qdrant_client
from baize.knowledge.models import KnowledgeChunkModel, KnowledgeDocumentModel
from baize.knowledge.schemas import (
    CitationResponse,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentResponse,
    SearchRequest,
    SearchResponse,
)
from baize.knowledge.service import KnowledgeBaseService, KnowledgeBaseServiceError, RetrievalService
from baize.llm.provider import ProviderFactory
from baize.user.deps import get_current_user
from baize.user.models import UserModel

_EXTENSION_TO_SOURCE_TYPE: dict[str, str] = {
    ".md": "markdown",
    ".txt": "raw_text",
    ".docx": "docx",
    ".pdf": "pdf",
}
_SUPPORTED_EXTENSIONS = sorted(_EXTENSION_TO_SOURCE_TYPE.keys())
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

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


@router.post(
    "/{kb_id}/documents/upload",
    response_model=KnowledgeDocumentResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
)
async def upload_document(
    kb_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    source_uri: str | None = Form(None),
    doc_metadata: str | None = Form(None),
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """Upload a file to a knowledge base via multipart/form-data.

    Performs synchronous pre-checks (format, size, encoding, encryption) and
    returns 202 immediately after creating a pending document. Ingestion is
    processed asynchronously in the background.

    Returns:
        202 Accepted + KnowledgeDocumentResponse with status='pending'.

    Raises:
        HTTPException: 415 if file extension is not .md/.txt/.docx/.pdf.
        HTTPException: 413 if file exceeds 10 MB.
        HTTPException: 400 if .md/.txt file is not UTF-8 encoded.
        HTTPException: 400 if .pdf file is encrypted.
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found or deleted.
    """
    logger.info(
        "POST /knowledge-bases/%s/documents/upload user_id=%s filename=%s",
        kb_id,
        current_user.id,
        file.filename,
    )

    suffix = Path(file.filename or "").suffix.lower()
    source_type = _EXTENSION_TO_SOURCE_TYPE.get(suffix)
    if source_type is None:
        raise HTTPException(
            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Supported formats: {', '.join(_SUPPORTED_EXTENSIONS)}"
            ),
        )

    raw_bytes = await file.read()

    if len(raw_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 10 MB limit.",
        )

    if suffix in {".md", ".txt"}:
        try:
            raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="File must be encoded in UTF-8. Please re-save the file with UTF-8 encoding.",
            )

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        if reader.is_encrypted:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Encrypted PDF files are not supported. Please decrypt the file before uploading.",
            )

    try:
        await svc.get(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc

    effective_title = title if title else Path(file.filename or "upload").stem

    parsed_metadata: dict | None = None
    if doc_metadata:
        try:
            parsed_metadata = json.loads(doc_metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON in doc_metadata: {exc}",
            )

    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    now = datetime.now(UTC)
    doc = KnowledgeDocumentModel(
        id=uuid.uuid4(),
        kb_id=kb_id,
        title=effective_title,
        source_type=source_type,
        source_uri=source_uri,
        content_hash=content_hash,
        doc_metadata=parsed_metadata if parsed_metadata is not None else {},
        status="pending",
        chunk_count=0,
        created_at=now,
        updated_at=now,
    )
    db.add(doc)
    await db.commit()

    background_tasks.add_task(
        _bg_ingest_file,
        doc_id=doc.id,
        kb_id=kb_id,
        raw_bytes=raw_bytes,
        source_type=source_type,
    )

    return KnowledgeDocumentResponse.model_validate(doc)


@router.get("/{kb_id}/documents", response_model=KnowledgeDocumentListResponse)
async def list_documents(
    kb_id: uuid.UUID,
    doc_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentListResponse:
    """List documents in a knowledge base with optional status filter.

    Returns:
        200 + KnowledgeDocumentListResponse with items and total.

    Raises:
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found or deleted.
    """
    logger.info(
        "GET /knowledge-bases/%s/documents user_id=%s status=%s",
        kb_id,
        current_user.id,
        doc_status,
    )
    try:
        await svc.get(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc

    base_where = [KnowledgeDocumentModel.kb_id == kb_id]
    if doc_status is not None:
        base_where.append(KnowledgeDocumentModel.status == doc_status)

    count_result = await db.execute(
        select(func.count()).select_from(KnowledgeDocumentModel).where(*base_where)
    )
    total: int = count_result.scalar_one()

    items_result = await db.execute(
        select(KnowledgeDocumentModel)
        .where(*base_where)
        .order_by(KnowledgeDocumentModel.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = items_result.scalars().all()

    return KnowledgeDocumentListResponse(
        items=[KnowledgeDocumentResponse.model_validate(doc) for doc in items],
        total=total,
    )


@router.get("/{kb_id}/documents/{doc_id}", response_model=KnowledgeDocumentResponse)
async def get_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """Return a single document belonging to the given knowledge base.

    Returns:
        200 + KnowledgeDocumentResponse.

    Raises:
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found, deleted, or document not found/in wrong KB.
    """
    logger.info(
        "GET /knowledge-bases/%s/documents/%s user_id=%s", kb_id, doc_id, current_user.id
    )
    try:
        await svc.get(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc

    result = await db.execute(
        select(KnowledgeDocumentModel).where(
            KnowledgeDocumentModel.id == doc_id,
            KnowledgeDocumentModel.kb_id == kb_id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found in knowledge base {kb_id}.",
        )
    return KnowledgeDocumentResponse.model_validate(doc)


@router.delete("/{kb_id}/documents/{doc_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Hard-delete a document and all its chunks from Postgres and Qdrant.

    Qdrant deletion failures are logged as warnings and do not block the
    Postgres deletion.

    Returns:
        204 No Content.

    Raises:
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found, deleted, or document not found/in wrong KB.
    """
    logger.info(
        "DELETE /knowledge-bases/%s/documents/%s user_id=%s", kb_id, doc_id, current_user.id
    )
    try:
        await svc.get(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc

    result = await db.execute(
        select(KnowledgeDocumentModel).where(
            KnowledgeDocumentModel.id == doc_id,
            KnowledgeDocumentModel.kb_id == kb_id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found in knowledge base {kb_id}.",
        )

    chunks_result = await db.execute(
        select(KnowledgeChunkModel).where(KnowledgeChunkModel.doc_id == doc_id)
    )
    chunks = chunks_result.scalars().all()

    point_ids = [str(c.qdrant_point_id) for c in chunks if c.qdrant_point_id is not None]
    if point_ids:
        collection_name = f"baize_kb_{kb_id.hex}"
        try:
            qdrant = await get_qdrant_client()
            await qdrant.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
        except Exception as exc:
            logger.warning(
                "Qdrant delete failed for document %s in collection %s (continuing): %s",
                doc_id,
                collection_name,
                exc,
            )

    await db.execute(delete(KnowledgeChunkModel).where(KnowledgeChunkModel.doc_id == doc_id))
    await db.execute(delete(KnowledgeDocumentModel).where(KnowledgeDocumentModel.id == doc_id))
    await db.commit()


async def _bg_ingest_file(
    *,
    doc_id: uuid.UUID,
    kb_id: uuid.UUID,
    raw_bytes: bytes,
    source_type: str,
) -> None:
    """Background task: convert a file and process it through the ingestion pipeline."""
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
                "Background file ingestion: KB %s not found for doc %s", kb_id, doc_id
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
            await ingestion_svc.process_pending_document_from_file(
                doc_id=doc_id,
                kb_id=kb_id,
                raw_bytes=raw_bytes,
                source_type=source_type,
            )
        except Exception:
            logger.exception("Background file ingestion failed for document %s", doc_id)


@router.post("/{kb_id}/search", response_model=SearchResponse)
async def search_knowledge_base(
    kb_id: uuid.UUID,
    body: SearchRequest,
    current_user: UserModel = Depends(get_current_user),
    svc: KnowledgeBaseService = Depends(get_knowledge_base_service),
    retrieval_svc: RetrievalService = Depends(get_retrieval_service),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Search a knowledge base using hybrid BM25 + vector retrieval.

    Returns:
        SearchResponse with query, ranked CitationResponse list, total, and latency_ms.

    Raises:
        HTTPException: 403 if KB belongs to another user.
        HTTPException: 404 if KB not found or deleted.
    """
    logger.info(
        "POST /knowledge-bases/%s/search user_id=%s", kb_id, current_user.id
    )

    try:
        await svc.get(kb_id=kb_id, user_id=current_user.id)
    except KnowledgeBaseServiceError as exc:
        raise _service_error_to_http(exc) from exc

    from baize.core.observability import TraceCollector

    tracer = TraceCollector(
        user_id=str(current_user.id),
        session_id=str(kb_id),
        agent_id="knowledge-search",
    )
    tracer.set_input(body.query)

    start = time.monotonic()
    chunks = await retrieval_svc.search(
        kb_id=kb_id,
        query=body.query,
        top_k=body.top_k,
        include_parents=body.include_parents,
        tracer=tracer,
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    doc_ids = {c.doc_id for c in chunks}
    doc_title_map: dict[uuid.UUID, str] = {}
    if doc_ids:
        rows_result = await db.execute(
            select(KnowledgeDocumentModel.id, KnowledgeDocumentModel.title).where(
                KnowledgeDocumentModel.id.in_(doc_ids)
            )
        )
        for row_doc_id, row_title in rows_result.all():
            doc_title_map[row_doc_id] = row_title

    results = [
        CitationResponse(
            doc_title=doc_title_map.get(c.doc_id, ""),
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            section_path=c.section_path,
            score=c.score,
            sources=c.sources,
        )
        for c in chunks
    ]

    tracer.finalize(output=f"{len(results)} results")

    return SearchResponse(
        query=body.query,
        results=results,
        total=len(results),
        latency_ms=latency_ms,
    )
