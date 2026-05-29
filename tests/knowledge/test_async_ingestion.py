"""Async ingestion unit tests (F-013).

Verifies IngestionService's background pipeline directly:
- process_pending_document: markdown/raw_text ingestion
- process_pending_document_from_file: docx/pdf file ingestion
- Error cases: scanned PDF, duplicate content, embedder failure, dim mismatch
"""

import json
import re
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.converters.exceptions import ScannedPdfError
from baize.knowledge.ingestion.exceptions import DocumentAlreadyExistsError
from baize.knowledge.ingestion.qdrant_client import VectorDimMismatchError
from baize.knowledge.ingestion.service import IngestionService
from baize.knowledge.models import KnowledgeBaseModel, KnowledgeChunkModel, KnowledgeDocumentModel
from sqlalchemy.sql.dml import Update as SQLAUpdate

# ─── Test stubs ────────────────────────────────────────────────────────────────


class _KbStub:
    """Minimal stand-in for KnowledgeBaseModel."""

    def __init__(self, kb_id: uuid.UUID, dim: int = 3) -> None:
        self.id = kb_id
        self.embedding_provider = "test-provider"
        self.embedding_model = "test-model"
        self.embedding_dim = dim
        self.status = "active"


class _DocStub:
    """Minimal stand-in for KnowledgeDocumentModel with mutable fields."""

    def __init__(
        self,
        doc_id: uuid.UUID,
        kb_id: uuid.UUID,
        source_type: str = "markdown",
    ) -> None:
        self.id = doc_id
        self.kb_id = kb_id
        self.title = "Test Doc"
        self.status = "pending"
        self.source_type = source_type
        self.doc_metadata: dict = {}
        self.content_hash: str | None = None
        self.chunk_count: int = 0
        self.error_message: str | None = None


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def kb_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def doc_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def kb_stub(kb_id: uuid.UUID) -> _KbStub:
    return _KbStub(kb_id)


@pytest.fixture
def doc_stub(doc_id: uuid.UUID, kb_id: uuid.UUID) -> _DocStub:
    return _DocStub(doc_id, kb_id)


def _empty_result() -> MagicMock:
    r = MagicMock()
    r.scalar_one_or_none.return_value = None
    r.scalars.return_value = MagicMock(all=MagicMock(return_value=[]))
    r.scalar_one.return_value = 0
    return r


_UPDATE_ATTRS = frozenset(["status", "error_message", "chunk_count"])


def _apply_update(stmt: SQLAUpdate, stub: _DocStub) -> None:
    """Extract UPDATE values via SQLAlchemy compilation and apply them to stub."""
    try:
        from sqlalchemy.dialects import sqlite as _sqlite
        compiled = stmt.compile(dialect=_sqlite.dialect())
        for raw_key, val in compiled.params.items():
            attr = re.sub(r"_\d+$", "", raw_key)
            if attr in _UPDATE_ATTRS and hasattr(stub, attr):
                setattr(stub, attr, val)
    except Exception:
        pass


def _make_session(kb_stub: _KbStub, doc_stub: _DocStub) -> AsyncMock:
    """Build a mock AsyncSession that applies UPDATE statements to doc_stub."""
    session = AsyncMock()

    async def _get(model_class, pk, **_kw):
        if model_class is KnowledgeBaseModel:
            return kb_stub
        if model_class is KnowledgeDocumentModel:
            return doc_stub
        return None

    async def _execute(stmt, *_args, **_kwargs):
        if isinstance(stmt, SQLAUpdate):
            _apply_update(stmt, doc_stub)
        return _empty_result()

    session.get = AsyncMock(side_effect=_get)
    session.execute = AsyncMock(side_effect=_execute)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def mock_session(kb_stub: _KbStub, doc_stub: _DocStub) -> AsyncMock:
    return _make_session(kb_stub, doc_stub)


@pytest.fixture
def mock_embedder() -> MagicMock:
    """KnowledgeEmbedder mock returning one dim=3 vector per input text."""
    embedder = MagicMock()
    embedder.embed_documents = AsyncMock(
        side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    )
    return embedder


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """AsyncQdrantClient mock where collection exists with matching dim=3."""
    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=True)
    vectors_mock = MagicMock()
    vectors_mock.size = 3
    col_info = MagicMock()
    col_info.config.params.vectors = vectors_mock
    qdrant.get_collection = AsyncMock(return_value=col_info)
    qdrant.upsert = AsyncMock()
    qdrant.delete = AsyncMock()
    return qdrant


@pytest.fixture
def svc(mock_session: AsyncMock, mock_embedder: MagicMock, mock_qdrant: MagicMock) -> IngestionService:
    return IngestionService(session=mock_session, embedder=mock_embedder, qdrant=mock_qdrant)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _added_chunks(session: AsyncMock) -> list[KnowledgeChunkModel]:
    """Extract KnowledgeChunkModel instances from session.add() calls."""
    return [
        c.args[0]
        for c in session.add.call_args_list
        if isinstance(c.args[0], KnowledgeChunkModel)
    ]


# ─── Success path tests ──────────────────────────────────────────────────────


async def test_md_ingestion_sets_ingested_with_chunks(svc, doc_stub, mock_session):
    """process_pending_document → status=ingested, source_type=markdown, chunks in DB."""
    doc_stub.source_type = "markdown"

    await svc.process_pending_document(
        doc_id=doc_stub.id,
        kb_id=doc_stub.kb_id,
        content="# Hello\n\nMarkdown content for the ingestion test.",
    )

    assert doc_stub.status == "ingested"
    assert doc_stub.source_type == "markdown"
    chunks = _added_chunks(mock_session)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.doc_id == doc_stub.id
        assert chunk.kb_id == doc_stub.kb_id


async def test_txt_ingestion_sets_ingested(svc, doc_stub, mock_session):
    """process_pending_document_from_file with raw_text → source_type preserved, status=ingested."""
    doc_stub.source_type = "raw_text"

    with patch("baize.knowledge.ingestion.service.convert_to_markdown") as mock_conv:
        mock_conv.return_value = ConvertedDocument(
            markdown="    Hello World.\n\n    Second paragraph.",
            warnings=[],
        )
        await svc.process_pending_document_from_file(
            doc_id=doc_stub.id,
            kb_id=doc_stub.kb_id,
            raw_bytes=b"Hello World.\n\nSecond paragraph.",
            source_type="raw_text",
        )

    assert doc_stub.status == "ingested"
    assert doc_stub.source_type == "raw_text"


async def test_docx_ingestion_chunks_have_section_path_from_headings(svc, doc_stub, mock_session):
    """process_pending_document_from_file for docx → chunks in DB, section_path from headings."""
    with patch("baize.knowledge.ingestion.service.convert_to_markdown") as mock_conv:
        mock_conv.return_value = ConvertedDocument(
            markdown="# Chapter One\n\nChapter intro.\n\n## Section A\n\nSection A body.",
            warnings=["Image discarded"],
        )
        await svc.process_pending_document_from_file(
            doc_id=doc_stub.id,
            kb_id=doc_stub.kb_id,
            raw_bytes=b"fake docx bytes",
            source_type="docx",
        )

    assert doc_stub.status == "ingested"
    chunks = _added_chunks(mock_session)
    assert len(chunks) >= 2

    # At least one chunk must carry a section_path derived from the heading tree
    section_paths = [
        json.loads(c.section_path)
        for c in chunks
        if c.section_path
    ]
    all_titles = [title for path in section_paths for title in path]
    assert any("Chapter One" in t for t in all_titles), (
        f"Expected 'Chapter One' in section_paths; got {section_paths}"
    )


async def test_pdf_with_outline_chunks_have_section_path(svc, doc_stub, mock_session):
    """process_pending_document_from_file for pdf with outline → section_path from outline titles."""
    with patch("baize.knowledge.ingestion.service.convert_to_markdown") as mock_conv:
        mock_conv.return_value = ConvertedDocument(
            markdown="# Introduction\n\nPDF intro text.\n\n## Methods\n\nMethods body.",
            warnings=["共 5 页，提取字符数 500"],
        )
        await svc.process_pending_document_from_file(
            doc_id=doc_stub.id,
            kb_id=doc_stub.kb_id,
            raw_bytes=b"fake pdf bytes",
            source_type="pdf",
        )

    assert doc_stub.status == "ingested"
    chunks = _added_chunks(mock_session)
    assert len(chunks) >= 2

    all_path_values = [
        title
        for c in chunks
        if c.section_path
        for title in json.loads(c.section_path)
    ]
    assert any("Introduction" in v or "Methods" in v for v in all_path_values), (
        f"Expected outline-derived section paths; got {all_path_values}"
    )


async def test_pdf_without_outline_ingested_with_warning(svc, doc_stub, mock_session):
    """PDF without outline → status=ingested, converter_warnings contains no-outline hint."""
    no_outline_warn = "该 PDF 无 outline（书签），已退化为纯文本输出"

    with patch("baize.knowledge.ingestion.service.convert_to_markdown") as mock_conv:
        mock_conv.return_value = ConvertedDocument(
            markdown="Plain text PDF content without any headings.",
            warnings=["共 3 页，提取字符数 100", no_outline_warn],
        )
        await svc.process_pending_document_from_file(
            doc_id=doc_stub.id,
            kb_id=doc_stub.kb_id,
            raw_bytes=b"fake pdf bytes",
            source_type="pdf",
        )

    assert doc_stub.status == "ingested"
    warnings = doc_stub.doc_metadata.get("converter_warnings", [])
    assert len(warnings) >= 1
    assert any("outline" in w or "大纲" in w for w in warnings), (
        f"Expected no-outline warning; got {warnings}"
    )


# ─── Failure path tests ──────────────────────────────────────────────────────


async def test_scanned_pdf_sets_failed_with_ocr_message(svc, doc_stub, mock_session):
    """ScannedPdfError → doc.status=failed, error_message contains '扫描版 PDF' or 'OCR'."""
    scanned_msg = "PDF 全文提取结果为空，可能是扫描版 PDF，请使用 OCR 工具处理后再导入"

    with patch("baize.knowledge.ingestion.service.convert_to_markdown") as mock_conv:
        mock_conv.side_effect = ScannedPdfError(scanned_msg)

        with pytest.raises(ScannedPdfError) as exc_info:
            await svc.process_pending_document_from_file(
                doc_id=doc_stub.id,
                kb_id=doc_stub.kb_id,
                raw_bytes=b"scanned pdf",
                source_type="pdf",
            )

    error_str = str(exc_info.value)
    assert "扫描版 PDF" in error_str or "OCR" in error_str

    assert doc_stub.status == "failed"
    assert doc_stub.error_message is not None
    assert "扫描版 PDF" in doc_stub.error_message or "OCR" in doc_stub.error_message
    mock_session.rollback.assert_called()


async def test_duplicate_file_content_sets_failed_with_existing_doc_id(
    svc, doc_stub, mock_session
):
    """Duplicate content detected after conversion → doc.status=failed, error_message has existing_doc_id."""
    existing_doc_id = uuid.uuid4()
    existing_doc = MagicMock()
    existing_doc.id = existing_doc_id

    execute_calls = [0]

    async def dup_execute(stmt, *_args, **_kw):
        execute_calls[0] += 1
        if isinstance(stmt, SQLAUpdate):
            _apply_update(stmt, doc_stub)
            return _empty_result()
        # First select call is the duplicate-check query
        if execute_calls[0] == 1:
            r = MagicMock()
            r.scalar_one_or_none.return_value = existing_doc
            return r
        return _empty_result()

    mock_session.execute = AsyncMock(side_effect=dup_execute)

    with patch("baize.knowledge.ingestion.service.convert_to_markdown") as mock_conv:
        mock_conv.return_value = ConvertedDocument(
            markdown="# Duplicate\n\nExactly the same content.", warnings=[]
        )
        with pytest.raises(DocumentAlreadyExistsError) as exc_info:
            await svc.process_pending_document_from_file(
                doc_id=doc_stub.id,
                kb_id=doc_stub.kb_id,
                raw_bytes=b"duplicate file",
                source_type="markdown",
            )

    error_str = str(exc_info.value)
    assert str(existing_doc_id) in error_str

    assert doc_stub.status == "failed"
    assert doc_stub.error_message is not None
    assert str(existing_doc_id) in doc_stub.error_message
    mock_session.rollback.assert_called()


async def test_embedder_exception_sets_failed_and_cleanup_called(
    svc, doc_stub, mock_session, mock_embedder
):
    """Embedder raises → doc.status=failed, cleanup_failed_document called, no chunks added."""
    mock_embedder.embed_documents = AsyncMock(
        side_effect=RuntimeError("Embedding service unavailable")
    )

    with patch.object(svc, "cleanup_failed_document", wraps=svc.cleanup_failed_document) as mock_cleanup:
        with pytest.raises(RuntimeError, match="Embedding service unavailable"):
            await svc.process_pending_document(
                doc_id=doc_stub.id,
                kb_id=doc_stub.kb_id,
                content="# Test\n\nContent that will fail during embedding.",
            )
        mock_cleanup.assert_called_once_with(doc_id=doc_stub.id)

    assert doc_stub.status == "failed"
    mock_session.rollback.assert_called()
    # Embedding fails before session.add(chunk), so no chunks should have been added
    assert len(_added_chunks(mock_session)) == 0


async def test_embedder_dim_mismatch_sets_failed_with_dim_mismatch_message(
    svc, doc_stub, mock_session, mock_embedder
):
    """Embedder returns wrong dim → doc.status=failed, error_message contains 'dim mismatch'."""
    # kb_stub.embedding_dim = 3; returning dim=5 triggers VectorDimMismatchError
    mock_embedder.embed_documents = AsyncMock(
        side_effect=lambda texts: [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]
    )

    with pytest.raises(VectorDimMismatchError, match="dim mismatch"):
        await svc.process_pending_document(
            doc_id=doc_stub.id,
            kb_id=doc_stub.kb_id,
            content="# Test\n\nContent for dimension mismatch scenario.",
        )

    assert doc_stub.status == "failed"
    assert doc_stub.error_message is not None
    assert "dim mismatch" in doc_stub.error_message
    mock_session.rollback.assert_called()
    # Dim mismatch is detected before session.add(chunk)
    assert len(_added_chunks(mock_session)) == 0
