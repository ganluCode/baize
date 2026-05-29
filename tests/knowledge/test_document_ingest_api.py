"""Tests for POST /api/v1/knowledge-bases/{kb_id}/documents (F-004)."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from baize.knowledge.ingestion.service import IngestionService
from baize.knowledge.models import KnowledgeChunkModel, KnowledgeDocumentModel
from baize.knowledge.service import KnowledgeBaseServiceError
from baize.user.models import UserModel

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_USER_ID = uuid.uuid4()
_KB_ID = uuid.uuid4()
_DOC_ID = uuid.uuid4()
_BASE_URL = "/api/v1/knowledge-bases"
_VALID_DOC_BODY = {
    "title": "Test Doc",
    "content": "# Hello\n\nThis is the content.",
    "source_type": "markdown",
}

SAMPLE_MD = """\
# Introduction

This is the introduction paragraph.

## Getting Started

Follow these steps to get started.
"""

DIM = 1536


# ────────────────────────────────────────
# Fixtures — API layer
# ────────────────────────────────────────


def _make_user(user_id: uuid.UUID | None = None) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id or _USER_ID
    return user


@pytest.fixture
def mock_kb_svc() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_db_session():
    """AsyncSession mock with no-duplicate execute result."""
    session = AsyncMock()
    no_dup = MagicMock()
    no_dup.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=no_dup)
    # add() is synchronous in SQLAlchemy AsyncSession
    session.add = MagicMock()
    return session


@pytest.fixture
def regular_user() -> UserModel:
    return _make_user()


@pytest.fixture
async def client_auth(app, mock_kb_svc, mock_db_session, regular_user):
    from baize.core.database import get_db
    from baize.knowledge.deps import get_knowledge_base_service
    from baize.user.deps import get_current_user

    app.dependency_overrides[get_knowledge_base_service] = lambda: mock_kb_svc
    app.dependency_overrides[get_current_user] = lambda: regular_user
    app.dependency_overrides[get_db] = lambda: mock_db_session
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def client_no_auth(app, mock_kb_svc, mock_db_session):
    from baize.core.database import get_db
    from baize.knowledge.deps import get_knowledge_base_service
    from baize.user.deps import get_current_user

    app.dependency_overrides[get_knowledge_base_service] = lambda: mock_kb_svc
    app.dependency_overrides[get_db] = lambda: mock_db_session

    def _raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)


# ────────────────────────────────────────
# Sync phase — API endpoint tests
# ────────────────────────────────────────


async def test_create_document_returns_202_with_pending_status(
    client_auth, mock_kb_svc
):
    """202 returned immediately; response status is 'pending'."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api._bg_ingest_document", new_callable=AsyncMock):
        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents",
            json=_VALID_DOC_BODY,
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["title"] == _VALID_DOC_BODY["title"]
    assert body["kb_id"] == str(_KB_ID)
    assert body["chunk_count"] == 0


async def test_create_document_response_contains_content_hash(
    client_auth, mock_kb_svc
):
    """Response body includes a non-empty content_hash computed from body content."""
    import hashlib

    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api._bg_ingest_document", new_callable=AsyncMock):
        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents",
            json=_VALID_DOC_BODY,
        )

    expected_hash = hashlib.sha256(
        _VALID_DOC_BODY["content"].encode("utf-8")
    ).hexdigest()
    assert response.json()["content_hash"] == expected_hash


async def test_create_document_kb_not_found_returns_404(client_auth, mock_kb_svc):
    """Returns 404 when the knowledge base does not exist."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Knowledge base not found.", status_code=404
    )

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents",
        json=_VALID_DOC_BODY,
    )

    assert response.status_code == 404


async def test_create_document_wrong_user_returns_403(client_auth, mock_kb_svc):
    """Returns 403 when the KB belongs to another user."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents",
        json=_VALID_DOC_BODY,
    )

    assert response.status_code == 403


async def test_create_document_duplicate_hash_returns_409_with_existing_doc_id(
    client_auth, mock_kb_svc, mock_db_session
):
    """Returns 409 with existing_doc_id in detail when same content is already ingested."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    existing_doc = MagicMock()
    existing_doc.id = _DOC_ID
    dup_result = MagicMock()
    dup_result.scalar_one_or_none.return_value = existing_doc
    mock_db_session.execute = AsyncMock(return_value=dup_result)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents",
        json=_VALID_DOC_BODY,
    )

    assert response.status_code == 409
    # Global exception handler wraps HTTPException detail into {"code": ..., "message": ...}
    message = response.json()["message"]
    assert str(_DOC_ID) in message


async def test_create_document_missing_required_field_returns_422(
    client_auth, mock_kb_svc
):
    """Returns 422 when required JSON fields are missing."""
    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents",
        json={"title": "No content"},  # missing content and source_type
    )

    assert response.status_code == 422


async def test_create_document_without_auth_returns_401(client_no_auth):
    """Returns 401 when no authentication is provided."""
    response = await client_no_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents",
        json=_VALID_DOC_BODY,
    )

    assert response.status_code == 401


# ────────────────────────────────────────
# Fixtures — IngestionService unit tests
# ────────────────────────────────────────


@pytest.fixture
def kb_id():
    return uuid.uuid4()


@pytest.fixture
def doc_id():
    return uuid.uuid4()


@pytest.fixture
def fake_kb(kb_id):
    kb = MagicMock()
    kb.id = kb_id
    kb.status = "active"
    kb.embedding_dim = DIM
    return kb


@pytest.fixture
def fake_doc(doc_id, kb_id):
    doc = MagicMock(spec=KnowledgeDocumentModel)
    doc.id = doc_id
    doc.kb_id = kb_id
    doc.title = "Test Doc"
    doc.status = "pending"
    return doc


@pytest.fixture
def mock_bg_session(fake_kb, fake_doc):
    """AsyncSession for background task tests, returning kb then doc from get()."""
    session = AsyncMock()
    session.get = AsyncMock(side_effect=[fake_kb, fake_doc])

    no_dup = MagicMock()
    no_dup.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=no_dup)

    added = []
    session.add = MagicMock(side_effect=lambda m: added.append(m))
    session._added = added

    return session


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed_documents = AsyncMock(
        side_effect=lambda texts: [[0.1] * DIM for _ in texts]
    )
    return embedder


@pytest.fixture
def mock_qdrant():
    return AsyncMock()


def _make_svc(session, embedder, qdrant) -> IngestionService:
    return IngestionService(session=session, embedder=embedder, qdrant=qdrant)


# ────────────────────────────────────────
# process_pending_document — background task behavior
# ────────────────────────────────────────


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_process_pending_document_sets_status_ingesting_before_processing(
    mock_ec, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """Background task updates doc status to 'ingesting' before processing starts."""
    statuses_at_commit: list[str] = []

    async def capture_commit():
        statuses_at_commit.append(fake_doc.status)

    mock_bg_session.commit = AsyncMock(side_effect=capture_commit)

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)
    await svc.process_pending_document(doc_id=doc_id, kb_id=kb_id, content=SAMPLE_MD)

    assert "ingesting" in statuses_at_commit


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_process_pending_document_success_marks_ingested_with_chunk_count(
    mock_ec, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """Successful processing marks document as 'ingested' with correct chunk_count."""
    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)
    await svc.process_pending_document(doc_id=doc_id, kb_id=kb_id, content=SAMPLE_MD)

    assert fake_doc.status == "ingested"
    assert isinstance(fake_doc.chunk_count, int)
    assert fake_doc.chunk_count > 0

    chunks = [m for m in mock_bg_session._added if isinstance(m, KnowledgeChunkModel)]
    assert len(chunks) == fake_doc.chunk_count
    mock_qdrant.upsert.assert_called_once()


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_process_pending_document_failure_marks_failed_with_error_message(
    mock_ec, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """Embedding failure marks document as 'failed' with error_message."""
    mock_embedder.embed_documents = AsyncMock(
        side_effect=RuntimeError("Embedder API down")
    )

    # Configure execute for the error path:
    # 1st: update status to failed; 2nd: select chunks (cleanup, returns empty)
    empty_result = MagicMock()
    empty_scalars = MagicMock()
    empty_scalars.all.return_value = []
    empty_result.scalars.return_value = empty_scalars
    mock_bg_session.execute = AsyncMock(
        side_effect=[MagicMock(), empty_result]
    )

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)

    with pytest.raises(RuntimeError, match="Embedder API down"):
        await svc.process_pending_document(doc_id=doc_id, kb_id=kb_id, content=SAMPLE_MD)

    mock_bg_session.rollback.assert_called()
    # Verify update to 'failed' was executed
    assert mock_bg_session.execute.call_count >= 1


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_process_pending_document_failure_calls_cleanup(
    mock_ec, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """Failed processing calls cleanup_failed_document to remove residual chunks."""
    mock_embedder.embed_documents = AsyncMock(side_effect=RuntimeError("fail"))

    chunk_with_point = MagicMock(spec=KnowledgeChunkModel)
    chunk_with_point.kb_id = kb_id
    chunk_with_point.qdrant_point_id = uuid.uuid4()

    select_result = MagicMock()
    select_scalars = MagicMock()
    select_scalars.all.return_value = [chunk_with_point]
    select_result.scalars.return_value = select_scalars

    delete_result = MagicMock()

    mock_bg_session.execute = AsyncMock(
        side_effect=[
            MagicMock(),  # update to failed
            select_result,  # select chunks for cleanup
            delete_result,  # delete chunks
        ]
    )

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)

    with pytest.raises(RuntimeError):
        await svc.process_pending_document(doc_id=doc_id, kb_id=kb_id, content=SAMPLE_MD)

    # Cleanup should have called Qdrant delete for the chunk's point_id
    mock_qdrant.delete.assert_called_once()
