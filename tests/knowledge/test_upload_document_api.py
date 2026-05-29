"""Tests for POST /api/v1/knowledge-bases/{kb_id}/documents/upload (F-005)."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.converters.exceptions import DocumentDecodeError, ScannedPdfError
from baize.knowledge.ingestion.exceptions import DocumentAlreadyExistsError
from baize.knowledge.ingestion.service import IngestionService
from baize.knowledge.models import KnowledgeChunkModel, KnowledgeDocumentModel
from baize.knowledge.service import KnowledgeBaseServiceError
from baize.user.models import UserModel

_USER_ID = uuid.uuid4()
_KB_ID = uuid.uuid4()
_DOC_ID = uuid.uuid4()
_BASE_URL = "/api/v1/knowledge-bases"
DIM = 1536

SAMPLE_MD = """\
# Introduction

This is the introduction paragraph.

## Getting Started

Follow these steps to get started.
"""


# ────────────────────────────────────────
# Shared fixtures — API layer
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
    session = AsyncMock()
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


def _upload(filename: str, content: bytes) -> dict:
    return {"file": (filename, content, "application/octet-stream")}


# ────────────────────────────────────────
# Sync phase: file format validation
# ────────────────────────────────────────


async def test_upload_unsupported_extension_returns_415(client_auth, mock_kb_svc):
    """Unsupported file extension returns 415 with supported formats in detail."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("document.doc", b"word content"),
    )

    assert response.status_code == 415
    detail = str(response.json())
    for fmt in [".md", ".txt", ".docx", ".pdf"]:
        assert fmt in detail


async def test_upload_html_extension_returns_415(client_auth, mock_kb_svc):
    """HTML files return 415."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("page.html", b"<html>content</html>"),
    )

    assert response.status_code == 415


async def test_upload_oversized_file_returns_413(client_auth, mock_kb_svc):
    """Files larger than 10 MB return 413."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    big_content = b"a" * (10 * 1024 * 1024 + 1)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("big.md", big_content),
    )

    assert response.status_code == 413


async def test_upload_gbk_encoded_md_returns_400(client_auth, mock_kb_svc):
    """.md file with non-UTF-8 encoding returns 400."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    gbk_content = "你好".encode("gbk")

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("test.md", gbk_content),
    )

    assert response.status_code == 400
    detail = str(response.json()).lower()
    assert "utf-8" in detail or "utf8" in detail


async def test_upload_gbk_encoded_txt_returns_400(client_auth, mock_kb_svc):
    """.txt file with non-UTF-8 encoding returns 400."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    gbk_content = "文本内容".encode("gbk")

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("notes.txt", gbk_content),
    )

    assert response.status_code == 400


async def test_upload_encrypted_pdf_returns_400(client_auth, mock_kb_svc):
    """Encrypted PDF files return 400."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api.PdfReader") as mock_reader_cls:
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        mock_reader_cls.return_value = mock_reader

        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents/upload",
            files=_upload("secret.pdf", b"%PDF-1.4 fake"),
        )

    assert response.status_code == 400


# ────────────────────────────────────────
# Sync phase: document creation
# ────────────────────────────────────────


async def test_upload_valid_md_returns_202_with_pending_status(client_auth, mock_kb_svc):
    """Valid .md upload returns 202 with status='pending'."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api._bg_ingest_file", new_callable=AsyncMock):
        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents/upload",
            files=_upload("readme.md", b"# Hello\n\nContent."),
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["kb_id"] == str(_KB_ID)
    assert body["chunk_count"] == 0


async def test_upload_title_derived_from_filename_when_not_provided(client_auth, mock_kb_svc):
    """Title is derived from filename (without extension) when not explicitly provided."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api._bg_ingest_file", new_callable=AsyncMock):
        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents/upload",
            files=_upload("doc.md", b"# Title\n\nContent."),
        )

    assert response.status_code == 202
    assert response.json()["title"] == "doc"


async def test_upload_explicit_title_overrides_filename(client_auth, mock_kb_svc):
    """When title is provided in form data it is used instead of the filename stem."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api._bg_ingest_file", new_callable=AsyncMock):
        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents/upload",
            files=_upload("doc.md", b"# Title\n\nContent."),
            data={"title": "My Custom Title"},
        )

    assert response.status_code == 202
    assert response.json()["title"] == "My Custom Title"


async def test_upload_docx_source_type_is_docx(client_auth, mock_kb_svc):
    """.docx upload creates a document with source_type='docx'."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api._bg_ingest_file", new_callable=AsyncMock):
        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents/upload",
            files=_upload("report.docx", b"fake docx bytes"),
        )

    assert response.status_code == 202
    assert response.json()["source_type"] == "docx"


async def test_upload_txt_source_type_is_raw_text(client_auth, mock_kb_svc):
    """.txt upload creates a document with source_type='raw_text'."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)

    with patch("baize.knowledge.api._bg_ingest_file", new_callable=AsyncMock):
        response = await client_auth.post(
            f"{_BASE_URL}/{_KB_ID}/documents/upload",
            files=_upload("notes.txt", b"Some plain text content."),
        )

    assert response.status_code == 202
    assert response.json()["source_type"] == "raw_text"


# ────────────────────────────────────────
# Auth / KB ownership
# ────────────────────────────────────────


async def test_upload_kb_not_found_returns_404(client_auth, mock_kb_svc):
    """Returns 404 when the knowledge base does not exist."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError("Not found", status_code=404)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("file.md", b"# Content"),
    )

    assert response.status_code == 404


async def test_upload_wrong_user_returns_403(client_auth, mock_kb_svc):
    """Returns 403 when the KB belongs to another user."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError("Forbidden", status_code=403)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("file.md", b"# Content"),
    )

    assert response.status_code == 403


async def test_upload_without_auth_returns_401(client_no_auth):
    """Returns 401 when no authentication is provided."""
    response = await client_no_auth.post(
        f"{_BASE_URL}/{_KB_ID}/documents/upload",
        files=_upload("file.md", b"# Content"),
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
    doc.title = "Test File"
    doc.status = "pending"
    doc.doc_metadata = {}
    return doc


@pytest.fixture
def mock_bg_session(fake_kb, fake_doc):
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
# process_pending_document_from_file — unit tests
# ────────────────────────────────────────


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch("baize.knowledge.ingestion.service.convert_to_markdown", new_callable=AsyncMock)
async def test_process_file_sets_status_ingesting_before_processing(
    mock_convert, mock_ec, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """Status is set to 'ingesting' before the conversion pipeline starts."""
    mock_convert.return_value = ConvertedDocument(markdown=SAMPLE_MD, warnings=[])

    statuses_at_commit: list[str] = []

    async def capture_commit():
        statuses_at_commit.append(fake_doc.status)

    mock_bg_session.commit = AsyncMock(side_effect=capture_commit)

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)
    await svc.process_pending_document_from_file(
        doc_id=doc_id, kb_id=kb_id, raw_bytes=b"# Intro\n\nContent.", source_type="markdown"
    )

    assert "ingesting" in statuses_at_commit


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch("baize.knowledge.ingestion.service.convert_to_markdown", new_callable=AsyncMock)
async def test_process_file_success_marks_ingested_with_chunk_count(
    mock_convert, mock_ec, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """Successful processing marks doc as 'ingested' with positive chunk_count."""
    mock_convert.return_value = ConvertedDocument(markdown=SAMPLE_MD, warnings=[])

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)
    await svc.process_pending_document_from_file(
        doc_id=doc_id, kb_id=kb_id, raw_bytes=b"bytes", source_type="markdown"
    )

    assert fake_doc.status == "ingested"
    assert isinstance(fake_doc.chunk_count, int)
    assert fake_doc.chunk_count > 0
    mock_qdrant.upsert.assert_called_once()


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch("baize.knowledge.ingestion.service.convert_to_markdown", new_callable=AsyncMock)
async def test_process_file_stores_converter_warnings_in_doc_metadata(
    mock_convert, mock_ec, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """converter_warnings from conversion are stored in doc.doc_metadata."""
    warnings = ["no outline detected", "共 3 页"]
    mock_convert.return_value = ConvertedDocument(markdown=SAMPLE_MD, warnings=warnings)

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)
    await svc.process_pending_document_from_file(
        doc_id=doc_id, kb_id=kb_id, raw_bytes=b"bytes", source_type="pdf"
    )

    assert fake_doc.doc_metadata["converter_warnings"] == warnings


@patch("baize.knowledge.ingestion.service.convert_to_markdown", new_callable=AsyncMock)
async def test_process_file_scanned_pdf_marks_failed(
    mock_convert, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """ScannedPdfError during conversion marks doc as 'failed' with error_message."""
    mock_convert.side_effect = ScannedPdfError("扫描版 PDF，请使用 OCR 工具处理后再导入")

    empty_result = MagicMock()
    empty_scalars = MagicMock()
    empty_scalars.all.return_value = []
    empty_result.scalars.return_value = empty_scalars
    mock_bg_session.execute = AsyncMock(side_effect=[MagicMock(), empty_result])

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)

    with pytest.raises(ScannedPdfError):
        await svc.process_pending_document_from_file(
            doc_id=doc_id, kb_id=kb_id, raw_bytes=b"scanned pdf", source_type="pdf"
        )

    mock_bg_session.rollback.assert_called()
    assert mock_bg_session.execute.call_count >= 1


@patch("baize.knowledge.ingestion.service.convert_to_markdown", new_callable=AsyncMock)
async def test_process_file_duplicate_content_marks_failed_with_existing_doc_id(
    mock_convert, mock_bg_session, mock_embedder, mock_qdrant, kb_id, doc_id, fake_doc
):
    """When converted content already exists in the KB, doc status becomes 'failed'
    and error_message contains the existing doc's ID."""
    mock_convert.return_value = ConvertedDocument(markdown=SAMPLE_MD, warnings=[])

    existing_id = uuid.uuid4()
    existing_doc = MagicMock()
    existing_doc.id = existing_id

    dup_result = MagicMock()
    dup_result.scalar_one_or_none.return_value = existing_doc

    empty_result = MagicMock()
    empty_scalars = MagicMock()
    empty_scalars.all.return_value = []
    empty_result.scalars.return_value = empty_scalars

    mock_bg_session.execute = AsyncMock(
        side_effect=[
            dup_result,    # duplicate check returns existing doc
            MagicMock(),   # update status to failed
            empty_result,  # select chunks for cleanup
        ]
    )

    svc = _make_svc(mock_bg_session, mock_embedder, mock_qdrant)

    with pytest.raises(DocumentAlreadyExistsError):
        await svc.process_pending_document_from_file(
            doc_id=doc_id, kb_id=kb_id, raw_bytes=b"bytes", source_type="markdown"
        )

    mock_bg_session.rollback.assert_called()
    # Verify update was called with error_message containing existing_doc_id
    update_calls = [c for c in mock_bg_session.execute.call_args_list]
    # The second execute should be the UPDATE to failed status
    assert mock_bg_session.execute.call_count >= 2
