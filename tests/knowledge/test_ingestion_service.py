"""Tests for IngestionService.ingest_markdown."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baize.knowledge.ingestion.exceptions import DocumentAlreadyExistsError
from baize.knowledge.ingestion.qdrant_client import VectorDimMismatchError
from baize.knowledge.ingestion.service import IngestionService
from baize.knowledge.models import KnowledgeChunkModel, KnowledgeDocumentModel

SAMPLE_MD = """\
# Introduction

This is the introduction paragraph.

## Getting Started

Follow these steps to get started.

### Installation

Run pip install to install the package.
"""

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

    no_dup_result = MagicMock()
    no_dup_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=no_dup_result)

    added = []
    session.add = MagicMock(side_effect=lambda model: added.append(model))
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


# ── KB validation ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_kb_not_found_raises(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    mock_session.get = AsyncMock(return_value=None)
    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(ValueError, match="not found"):
        await svc.ingest_markdown(kb_id=kb_id, title="T", content="c")


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_kb_not_active_raises(mock_ec, fake_kb, mock_session, mock_embedder, mock_qdrant, kb_id):
    fake_kb.status = "archived"
    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(ValueError, match="not active"):
        await svc.ingest_markdown(kb_id=kb_id, title="T", content="c")


# ── Duplicate detection ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_duplicate_document_raises(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    dup_result = MagicMock()
    dup_result.scalar_one_or_none.return_value = MagicMock()
    mock_session.execute = AsyncMock(return_value=dup_result)

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(DocumentAlreadyExistsError):
        await svc.ingest_markdown(kb_id=kb_id, title="T", content="dup")


# ── Content hash ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_content_hash_is_sha256_of_content(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    content = "deterministic content for hashing"
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_markdown(kb_id=kb_id, title="T", content=content)

    docs = [m for m in mock_session._added if isinstance(m, KnowledgeDocumentModel)]
    assert len(docs) == 1
    assert docs[0].content_hash == expected_hash


# ── Happy path ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_happy_path_returns_doc_id_and_marks_ingested(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    doc_id = await svc.ingest_markdown(kb_id=kb_id, title="Doc", content=SAMPLE_MD)

    assert isinstance(doc_id, uuid.UUID)

    docs = [m for m in mock_session._added if isinstance(m, KnowledgeDocumentModel)]
    assert len(docs) == 1
    assert docs[0].status == "ingested"
    assert docs[0].chunk_count > 0

    chunks = [m for m in mock_session._added if isinstance(m, KnowledgeChunkModel)]
    assert len(chunks) == docs[0].chunk_count

    mock_qdrant.upsert.assert_called_once()


# ── Batch embedding ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_embed_batch_size_not_exceed_32(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    sections = [f"## Section {i}\n\nParagraph A of {i}.\n\nParagraph B of {i}." for i in range(20)]
    big_content = "\n\n".join(sections)

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_markdown(kb_id=kb_id, title="Big", content=big_content)

    assert mock_embedder.embed_documents.call_count >= 2
    for call in mock_embedder.embed_documents.call_args_list:
        batch = call[0][0]
        assert len(batch) <= 32


# ── Dimension mismatch ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_dim_mismatch_raises_and_no_chunks_inserted(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    mock_embedder.embed_documents = AsyncMock(
        side_effect=lambda texts: [[0.1] * 512 for _ in texts]
    )

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(VectorDimMismatchError):
        await svc.ingest_markdown(kb_id=kb_id, title="T", content=SAMPLE_MD)

    chunks = [m for m in mock_session._added if isinstance(m, KnowledgeChunkModel)]
    assert len(chunks) == 0


# ── Parent-child chunk mapping ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_parent_chunk_id_mapped_correctly(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_markdown(kb_id=kb_id, title="Doc", content=SAMPLE_MD)

    chunks = [m for m in mock_session._added if isinstance(m, KnowledgeChunkModel)]
    chunk_ids = {c.id for c in chunks}

    for chunk in chunks:
        if chunk.parent_chunk_id is not None:
            assert chunk.parent_chunk_id in chunk_ids


# ── Qdrant point ID backfill ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_qdrant_point_id_backfilled(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    svc = _make_service(mock_session, mock_embedder, mock_qdrant)
    await svc.ingest_markdown(kb_id=kb_id, title="Doc", content=SAMPLE_MD)

    chunks = [m for m in mock_session._added if isinstance(m, KnowledgeChunkModel)]
    for chunk in chunks:
        assert chunk.qdrant_point_id is not None
        assert isinstance(chunk.qdrant_point_id, uuid.UUID)


# ── Failure handling ──


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_embedder_failure_marks_doc_failed(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    mock_embedder.embed_documents = AsyncMock(side_effect=RuntimeError("API error"))

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(RuntimeError, match="API error"):
        await svc.ingest_markdown(kb_id=kb_id, title="T", content=SAMPLE_MD)

    mock_session.rollback.assert_called()
    assert mock_session.execute.call_count >= 2
    assert mock_session.commit.call_count >= 2


@patch("baize.knowledge.ingestion.service.ensure_collection", new_callable=AsyncMock)
async def test_qdrant_failure_preserves_committed_chunks(mock_ec, mock_session, mock_embedder, mock_qdrant, kb_id):
    mock_qdrant.upsert = AsyncMock(side_effect=RuntimeError("Qdrant down"))

    svc = _make_service(mock_session, mock_embedder, mock_qdrant)

    with pytest.raises(RuntimeError, match="Qdrant down"):
        await svc.ingest_markdown(kb_id=kb_id, title="T", content=SAMPLE_MD)

    chunks = [m for m in mock_session._added if isinstance(m, KnowledgeChunkModel)]
    assert len(chunks) > 0
