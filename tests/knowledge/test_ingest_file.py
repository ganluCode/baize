"""Tests for IngestionService.ingest_file (F-010, F-019)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baize.knowledge.ingestion.converters import (
    DocumentDecodeError,
    ScannedPdfError,
    UnsupportedFormatError,
)
from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.service import IngestionService
from baize.knowledge.models import KnowledgeChunkModel, KnowledgeDocumentModel
from tests.knowledge.fixtures.conftest import sample_docx_bytes, sample_pdf_bytes

DIM = 1536


@pytest.fixture
def kb_id():
    return uuid.uuid4()


@pytest.fixture
def fake_kb(kb_id):
    kb = MagicMock()
    kb.id = kb_id
    kb.status = "active"
    kb.embedding_dim = DIM
    return kb


@pytest.fixture
def mock_session(fake_kb):
    session = AsyncMock()
    session.get = AsyncMock(return_value=fake_kb)
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


def _make_service(session, embedder, qdrant):
    return IngestionService(session=session, embedder=embedder, qdrant=qdrant)


# ── converter_warnings propagated into doc_metadata ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_converter_warnings_written_to_doc_metadata(
    mock_convert, mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id
):
    warnings = ["no outline detected", "2 pages, 500 chars"]
    mock_convert.return_value = ConvertedDocument(
        markdown="# Title\n\nSome content.", warnings=warnings
    )

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_file(
        kb_id=kb_id,
        title="Test",
        raw_bytes=b"fake pdf bytes",
        source_type="pdf",
    )

    from baize.knowledge.models import KnowledgeDocumentModel

    docs = [m for m in mock_session._added if isinstance(m, KnowledgeDocumentModel)]
    assert len(docs) == 1
    assert docs[0].doc_metadata["converter_warnings"] == warnings


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_empty_warnings_written_as_empty_list(
    mock_convert, mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id
):
    mock_convert.return_value = ConvertedDocument(
        markdown="# Title\n\nContent.", warnings=[]
    )

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_file(
        kb_id=kb_id,
        title="Test",
        raw_bytes=b"# Title\n\nContent.",
        source_type="markdown",
    )

    from baize.knowledge.models import KnowledgeDocumentModel

    docs = [m for m in mock_session._added if isinstance(m, KnowledgeDocumentModel)]
    assert docs[0].doc_metadata["converter_warnings"] == []


# ── caller-supplied doc_metadata is preserved ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_existing_doc_metadata_preserved(
    mock_convert, mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id
):
    mock_convert.return_value = ConvertedDocument(
        markdown="# Title\n\nSome content.", warnings=["w1"]
    )

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_file(
        kb_id=kb_id,
        title="Test",
        raw_bytes=b"bytes",
        source_type="pdf",
        doc_metadata={"author": "Alice"},
    )

    from baize.knowledge.models import KnowledgeDocumentModel

    docs = [m for m in mock_session._added if isinstance(m, KnowledgeDocumentModel)]
    meta = docs[0].doc_metadata
    assert meta["author"] == "Alice"
    assert meta["converter_warnings"] == ["w1"]


# ── convert_to_markdown is called with correct args ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_convert_to_markdown_called_with_correct_args(
    mock_convert, mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id
):
    raw = b"some docx bytes"
    mock_convert.return_value = ConvertedDocument(
        markdown="# Title\n\nContent.", warnings=[]
    )

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_file(
        kb_id=kb_id,
        title="Doc",
        raw_bytes=raw,
        source_type="docx",
    )

    mock_convert.assert_called_once_with(source_type="docx", raw_bytes=raw)


# ── converter errors propagate unchanged ──


@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_unsupported_format_error_propagates(
    mock_convert, mock_session, mock_embedder, mock_qdrant, kb_id
):
    mock_convert.side_effect = UnsupportedFormatError("Unsupported: html")

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(UnsupportedFormatError):
        await svc.ingest_file(
            kb_id=kb_id,
            title="T",
            raw_bytes=b"<html>",
            source_type="html",
        )


@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_scanned_pdf_error_propagates(
    mock_convert, mock_session, mock_embedder, mock_qdrant, kb_id
):
    mock_convert.side_effect = ScannedPdfError("扫描版 PDF，请使用 OCR")

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(ScannedPdfError):
        await svc.ingest_file(
            kb_id=kb_id,
            title="T",
            raw_bytes=b"pdf bytes",
            source_type="pdf",
        )


@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_document_decode_error_propagates(
    mock_convert, mock_session, mock_embedder, mock_qdrant, kb_id
):
    mock_convert.side_effect = DocumentDecodeError("加密 PDF")

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(DocumentDecodeError):
        await svc.ingest_file(
            kb_id=kb_id,
            title="T",
            raw_bytes=b"encrypted pdf",
            source_type="pdf",
        )


# ── return value is a UUID ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
@patch(
    "baize.knowledge.ingestion.service.convert_to_markdown",
    new_callable=AsyncMock,
)
async def test_returns_uuid(
    mock_convert, mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id
):
    mock_convert.return_value = ConvertedDocument(
        markdown="# Title\n\nContent.", warnings=[]
    )

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    result = await svc.ingest_file(
        kb_id=kb_id,
        title="Doc",
        raw_bytes=b"bytes",
        source_type="markdown",
    )

    assert isinstance(result, uuid.UUID)


# ── real converter chain with fixture bytes (F-019) ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_ingest_file_docx_real_converter_writes_chunks_and_warnings(
    mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id
):
    """传入真实 .docx 字节，通过真实 mammoth converter，chunks 落库且 doc_metadata 含 converter_warnings 键。"""
    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    doc_id = await svc.ingest_file(
        kb_id=kb_id,
        title="Sample Docx",
        raw_bytes=sample_docx_bytes(),
        source_type="docx",
    )

    assert isinstance(doc_id, uuid.UUID)

    docs = [m for m in mock_session._added if isinstance(m, KnowledgeDocumentModel)]
    assert len(docs) == 1
    assert "converter_warnings" in docs[0].doc_metadata

    chunks = [m for m in mock_session._added if isinstance(m, KnowledgeChunkModel)]
    assert len(chunks) > 0


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_ingest_file_pdf_real_converter_writes_chunks(
    mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id
):
    """传入真实 .pdf 字节，通过真实 pypdf converter，chunks 落库。"""
    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    doc_id = await svc.ingest_file(
        kb_id=kb_id,
        title="Sample PDF",
        raw_bytes=sample_pdf_bytes(),
        source_type="pdf",
    )

    assert isinstance(doc_id, uuid.UUID)

    docs = [m for m in mock_session._added if isinstance(m, KnowledgeDocumentModel)]
    assert len(docs) == 1
    assert "converter_warnings" in docs[0].doc_metadata

    chunks = [m for m in mock_session._added if isinstance(m, KnowledgeChunkModel)]
    assert len(chunks) > 0
