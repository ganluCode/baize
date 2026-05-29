"""Tests for Document query and delete API endpoints (F-006).

Covers:
  GET  /{kb_id}/documents            — list documents
  GET  /{kb_id}/documents/{doc_id}   — get single document
  DELETE /{kb_id}/documents/{doc_id} — hard delete with Qdrant cleanup
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from baize.knowledge.models import KnowledgeChunkModel, KnowledgeDocumentModel
from baize.knowledge.schemas import KnowledgeDocumentListResponse, KnowledgeDocumentResponse
from baize.knowledge.service import KnowledgeBaseServiceError
from baize.user.models import UserModel

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_USER_ID = uuid.uuid4()
_KB_ID = uuid.uuid4()
_DOC_ID = uuid.uuid4()
_BASE_URL = "/api/v1/knowledge-bases"


def _make_user(user_id: uuid.UUID | None = None) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id or _USER_ID
    return user


def _make_fake_doc(
    doc_id: uuid.UUID | None = None,
    kb_id: uuid.UUID | None = None,
    status: str = "ingested",
) -> KnowledgeDocumentModel:
    doc = MagicMock(spec=KnowledgeDocumentModel)
    doc.id = doc_id or _DOC_ID
    doc.kb_id = kb_id or _KB_ID
    doc.title = "Test Doc"
    doc.source_type = "markdown"
    doc.source_uri = None
    doc.content_hash = "abc123def456"
    doc.doc_metadata = {}
    doc.chunk_count = 3
    doc.status = status
    doc.error_message = None
    doc.created_at = _NOW
    doc.updated_at = _NOW
    return doc


def _make_fake_chunk(
    doc_id: uuid.UUID | None = None,
    kb_id: uuid.UUID | None = None,
    qdrant_point_id: uuid.UUID | None = None,
) -> KnowledgeChunkModel:
    chunk = MagicMock(spec=KnowledgeChunkModel)
    chunk.id = uuid.uuid4()
    chunk.doc_id = doc_id or _DOC_ID
    chunk.kb_id = kb_id or _KB_ID
    chunk.qdrant_point_id = qdrant_point_id or uuid.uuid4()
    return chunk


# ────────────────────────────────────────
# Fixtures — shared
# ────────────────────────────────────────


@pytest.fixture
def mock_kb_svc() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def regular_user() -> UserModel:
    return _make_user()


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    session.add = MagicMock()
    return session


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
# GET /{kb_id}/documents — list documents
# ────────────────────────────────────────


async def test_list_documents_returns_200_with_items_and_total(
    client_auth, mock_kb_svc, mock_db_session
):
    """Returns 200 with items list and total count."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    fake_doc = _make_fake_doc()

    count_result = MagicMock()
    count_result.scalar_one.return_value = 1

    items_result = MagicMock()
    items_scalars = MagicMock()
    items_scalars.all.return_value = [fake_doc]
    items_result.scalars.return_value = items_scalars

    mock_db_session.execute = AsyncMock(side_effect=[count_result, items_result])

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == str(_DOC_ID)
    assert body["items"][0]["kb_id"] == str(_KB_ID)


async def test_list_documents_returns_empty_when_no_documents(
    client_auth, mock_kb_svc, mock_db_session
):
    """Returns 200 with empty items and total=0 when KB has no documents."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    count_result = MagicMock()
    count_result.scalar_one.return_value = 0

    items_result = MagicMock()
    items_scalars = MagicMock()
    items_scalars.all.return_value = []
    items_result.scalars.return_value = items_scalars

    mock_db_session.execute = AsyncMock(side_effect=[count_result, items_result])

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []


async def test_list_documents_filters_by_status(
    client_auth, mock_kb_svc, mock_db_session
):
    """Status query parameter is applied to the query (endpoint completes without error)."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    count_result = MagicMock()
    count_result.scalar_one.return_value = 1

    fake_doc = _make_fake_doc(status="pending")
    items_result = MagicMock()
    items_scalars = MagicMock()
    items_scalars.all.return_value = [fake_doc]
    items_result.scalars.return_value = items_scalars

    mock_db_session.execute = AsyncMock(side_effect=[count_result, items_result])

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents?status=pending")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["status"] == "pending"


async def test_list_documents_kb_not_found_returns_404(client_auth, mock_kb_svc):
    """Returns 404 when the knowledge base does not exist."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Knowledge base not found.", status_code=404
    )

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents")

    assert response.status_code == 404


async def test_list_documents_wrong_user_returns_403(client_auth, mock_kb_svc):
    """Returns 403 when the KB belongs to another user."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents")

    assert response.status_code == 403


async def test_list_documents_without_auth_returns_401(client_no_auth):
    """Returns 401 when no authentication is provided."""
    response = await client_no_auth.get(f"{_BASE_URL}/{_KB_ID}/documents")

    assert response.status_code == 401


# ────────────────────────────────────────
# GET /{kb_id}/documents/{doc_id} — single document
# ────────────────────────────────────────


async def test_get_document_returns_200(client_auth, mock_kb_svc, mock_db_session):
    """Returns 200 with document data when doc exists in KB."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    fake_doc = _make_fake_doc()
    result = MagicMock()
    result.scalar_one_or_none.return_value = fake_doc
    mock_db_session.execute = AsyncMock(return_value=result)

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(_DOC_ID)
    assert body["kb_id"] == str(_KB_ID)
    assert body["status"] == "ingested"


async def test_get_document_not_found_returns_404(client_auth, mock_kb_svc, mock_db_session):
    """Returns 404 when doc does not exist in the KB."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=result)

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 404


async def test_get_document_kb_not_found_returns_404(client_auth, mock_kb_svc):
    """Returns 404 when KB does not exist."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Knowledge base not found.", status_code=404
    )

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 404


async def test_get_document_wrong_user_returns_403(client_auth, mock_kb_svc):
    """Returns 403 when KB belongs to another user."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 403


# ────────────────────────────────────────
# DELETE /{kb_id}/documents/{doc_id} — hard delete
# ────────────────────────────────────────


async def test_delete_document_returns_204(client_auth, mock_kb_svc, mock_db_session):
    """Returns 204 No Content after successful deletion."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    fake_doc = _make_fake_doc()
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = fake_doc

    # No chunks for simplicity
    chunks_result = MagicMock()
    chunks_scalars = MagicMock()
    chunks_scalars.all.return_value = []
    chunks_result.scalars.return_value = chunks_scalars

    mock_db_session.execute = AsyncMock(
        side_effect=[doc_result, chunks_result, MagicMock(), MagicMock()]
    )

    response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 204


async def test_delete_document_physically_deletes_postgres_records(
    client_auth, mock_kb_svc, mock_db_session
):
    """Postgres chunks and document are deleted via execute calls."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    fake_doc = _make_fake_doc()
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = fake_doc

    chunks_result = MagicMock()
    chunks_scalars = MagicMock()
    chunks_scalars.all.return_value = []
    chunks_result.scalars.return_value = chunks_scalars

    delete_chunks_result = MagicMock()
    delete_doc_result = MagicMock()

    mock_db_session.execute = AsyncMock(
        side_effect=[doc_result, chunks_result, delete_chunks_result, delete_doc_result]
    )

    await client_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    # 4 execute calls: select doc, select chunks, delete chunks, delete doc
    assert mock_db_session.execute.call_count == 4
    mock_db_session.commit.assert_called()


async def test_delete_document_calls_qdrant_delete_for_chunks_with_point_ids(
    client_auth, mock_kb_svc, mock_db_session
):
    """Qdrant delete is called with the point IDs from chunks."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    fake_doc = _make_fake_doc()
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = fake_doc

    point_id = uuid.uuid4()
    fake_chunk = _make_fake_chunk(qdrant_point_id=point_id)

    chunks_result = MagicMock()
    chunks_scalars = MagicMock()
    chunks_scalars.all.return_value = [fake_chunk]
    chunks_result.scalars.return_value = chunks_scalars

    mock_db_session.execute = AsyncMock(
        side_effect=[doc_result, chunks_result, MagicMock(), MagicMock()]
    )

    mock_qdrant = AsyncMock()

    with patch("baize.knowledge.api.get_qdrant_client", new_callable=AsyncMock) as mock_get_qdrant:
        mock_get_qdrant.return_value = mock_qdrant
        response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 204
    mock_qdrant.delete.assert_called_once()
    call_kwargs = mock_qdrant.delete.call_args
    assert call_kwargs.kwargs["collection_name"] == f"baize_kb_{_KB_ID.hex}"


async def test_delete_document_qdrant_failure_does_not_block_postgres_delete(
    client_auth, mock_kb_svc, mock_db_session
):
    """Qdrant failure is logged as warning and does not prevent Postgres deletion."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    fake_doc = _make_fake_doc()
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = fake_doc

    point_id = uuid.uuid4()
    fake_chunk = _make_fake_chunk(qdrant_point_id=point_id)

    chunks_result = MagicMock()
    chunks_scalars = MagicMock()
    chunks_scalars.all.return_value = [fake_chunk]
    chunks_result.scalars.return_value = chunks_scalars

    mock_db_session.execute = AsyncMock(
        side_effect=[doc_result, chunks_result, MagicMock(), MagicMock()]
    )

    mock_qdrant = AsyncMock()
    mock_qdrant.delete = AsyncMock(side_effect=RuntimeError("Qdrant unavailable"))

    with patch("baize.knowledge.api.get_qdrant_client", new_callable=AsyncMock) as mock_get_qdrant:
        mock_get_qdrant.return_value = mock_qdrant
        response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    # Despite Qdrant failure, Postgres deletion still succeeds
    assert response.status_code == 204
    mock_db_session.commit.assert_called()


async def test_delete_document_not_found_returns_404(client_auth, mock_kb_svc, mock_db_session):
    """Returns 404 when doc does not exist in the KB."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=result)

    response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 404


async def test_delete_document_kb_not_found_returns_404(client_auth, mock_kb_svc):
    """Returns 404 when KB does not exist."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Knowledge base not found.", status_code=404
    )

    response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 404


async def test_delete_document_wrong_user_returns_403(client_auth, mock_kb_svc):
    """Returns 403 when KB belongs to another user."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )

    response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 403


async def test_delete_document_without_auth_returns_401(client_no_auth):
    """Returns 401 when no authentication is provided."""
    response = await client_no_auth.delete(f"{_BASE_URL}/{_KB_ID}/documents/{_DOC_ID}")

    assert response.status_code == 401
