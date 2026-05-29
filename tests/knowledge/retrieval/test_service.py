"""Unit tests for RetrievalService — mocked KB, embedder, retriever, expand_parents."""

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baize.knowledge.retrieval.exceptions import (
    KnowledgeBaseNotActiveError,
    VectorDimMismatchError,
)
from baize.knowledge.retrieval.types import ScoredChunk


def _make_kb(status: str = "active") -> MagicMock:
    kb = MagicMock()
    kb.id = uuid.uuid4()
    kb.status = status
    kb.embedding_provider = "openai"
    kb.embedding_model = "text-embedding-3-small"
    kb.embedding_dim = 1536
    return kb


def _make_chunk(chunk_id: uuid.UUID, *, kb_id: uuid.UUID | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id=uuid.uuid4(),
        kb_id=kb_id or uuid.uuid4(),
        content="test content",
        section_path=[],
        level=1,
        parent_chunk_id=None,
        score=0.8,
        rank=1,
        sources=["bm25", "vector"],
        bm25_rank=1,
        vector_rank=1,
        bm25_score=0.8,
        vector_score=0.8,
    )


def _make_service(kb: MagicMock, embedder_factory=None):
    """Build a RetrievalService with mocked dependencies."""
    from baize.knowledge.service import RetrievalService

    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=kb)

    qdrant_client = AsyncMock()

    if embedder_factory is None:
        embedder = MagicMock()
        embedder_factory = MagicMock(return_value=embedder)

    return RetrievalService(
        db_session=db_session,
        qdrant_client=qdrant_client,
        embedder_factory=embedder_factory,
    ), db_session, qdrant_client, embedder_factory


# ── KB 状态校验 ──


async def test_inactive_kb_raises_not_active_error():
    """KB status=deleted must raise KnowledgeBaseNotActiveError without calling embedder."""
    kb = _make_kb(status="deleted")
    service, _, _, embedder_factory = _make_service(kb)

    with pytest.raises(KnowledgeBaseNotActiveError):
        await service.search(kb_id=uuid.uuid4(), query="test")

    embedder_factory.assert_not_called()


async def test_pending_kb_raises_not_active_error():
    """KB status=pending must also raise KnowledgeBaseNotActiveError."""
    kb = _make_kb(status="pending")
    service, _, _, _ = _make_service(kb)

    with pytest.raises(KnowledgeBaseNotActiveError):
        await service.search(kb_id=uuid.uuid4(), query="test")


async def test_not_found_kb_raises_not_active_error():
    """Missing KB (get returns None) must raise KnowledgeBaseNotActiveError."""
    from baize.knowledge.service import RetrievalService

    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=None)
    service = RetrievalService(
        db_session=db_session,
        qdrant_client=AsyncMock(),
        embedder_factory=MagicMock(),
    )

    with pytest.raises(KnowledgeBaseNotActiveError):
        await service.search(kb_id=uuid.uuid4(), query="test")


# ── embedder_factory 异常传播 ──


async def test_embedder_factory_dim_mismatch_propagates():
    """VectorDimMismatchError raised by embedder_factory must propagate."""
    kb = _make_kb(status="active")
    embedder_factory = MagicMock(side_effect=VectorDimMismatchError("dim mismatch"))
    service, _, _, _ = _make_service(kb, embedder_factory=embedder_factory)

    with pytest.raises(VectorDimMismatchError):
        await service.search(kb_id=uuid.uuid4(), query="test")


# ── top_k 默认值 ──


async def test_top_k_none_uses_settings_default():
    """When top_k=None, the actual top_k passed to HybridRetriever equals settings default."""
    from baize.knowledge.service import RetrievalService

    kb = _make_kb(status="active")
    kb_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    chunk_id = uuid.uuid4()

    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=kb)
    embedder_factory = MagicMock(return_value=MagicMock())

    service = RetrievalService(
        db_session=db_session,
        qdrant_client=AsyncMock(),
        embedder_factory=embedder_factory,
    )

    captured_top_k = {}

    async def fake_hybrid_search(*, kb_id, query, top_k, candidate_limit):
        captured_top_k["top_k"] = top_k
        return [_make_chunk(chunk_id, kb_id=kb_id)]

    with patch("baize.knowledge.service.HybridRetriever") as mock_cls:
        instance = AsyncMock()
        instance.search = fake_hybrid_search
        mock_cls.return_value = instance

        await service.search(kb_id=kb_id, query="test")

    from baize.core.config import settings

    assert captured_top_k["top_k"] == settings.knowledge_default_top_k


async def test_explicit_top_k_is_forwarded():
    """When top_k=5, HybridRetriever receives top_k=5."""
    kb = _make_kb(status="active")
    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=kb)

    from baize.knowledge.service import RetrievalService

    service = RetrievalService(
        db_session=db_session,
        qdrant_client=AsyncMock(),
        embedder_factory=MagicMock(return_value=MagicMock()),
    )

    captured = {}

    async def fake_search(*, kb_id, query, top_k, candidate_limit):
        captured["top_k"] = top_k
        return []

    with patch("baize.knowledge.service.HybridRetriever") as mock_cls:
        instance = AsyncMock()
        instance.search = fake_search
        mock_cls.return_value = instance

        await service.search(kb_id=uuid.uuid4(), query="test", top_k=5)

    assert captured["top_k"] == 5


# ── include_parents ──


async def test_include_parents_true_calls_expand_parents():
    """include_parents=True must invoke expand_parents with the retrieval results."""
    kb = _make_kb(status="active")
    kb_id = uuid.uuid4()
    chunk = _make_chunk(uuid.uuid4(), kb_id=kb_id)

    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=kb)

    from baize.knowledge.service import RetrievalService

    service = RetrievalService(
        db_session=db_session,
        qdrant_client=AsyncMock(),
        embedder_factory=MagicMock(return_value=MagicMock()),
    )

    with (
        patch("baize.knowledge.service.HybridRetriever") as mock_hybrid,
        patch("baize.knowledge.service.expand_parents", new_callable=AsyncMock) as mock_expand,
    ):
        hybrid_instance = AsyncMock()
        hybrid_instance.search = AsyncMock(return_value=[chunk])
        mock_hybrid.return_value = hybrid_instance

        parent_chunk = _make_chunk(uuid.uuid4(), kb_id=kb_id)
        mock_expand.return_value = [chunk, parent_chunk]

        results = await service.search(kb_id=kb_id, query="test", include_parents=True)

    mock_expand.assert_called_once()
    assert len(results) == 2


async def test_include_parents_false_does_not_call_expand_parents():
    """include_parents=False must not invoke expand_parents."""
    kb = _make_kb(status="active")

    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=kb)

    from baize.knowledge.service import RetrievalService

    service = RetrievalService(
        db_session=db_session,
        qdrant_client=AsyncMock(),
        embedder_factory=MagicMock(return_value=MagicMock()),
    )

    with (
        patch("baize.knowledge.service.HybridRetriever") as mock_hybrid,
        patch("baize.knowledge.service.expand_parents", new_callable=AsyncMock) as mock_expand,
    ):
        hybrid_instance = AsyncMock()
        hybrid_instance.search = AsyncMock(return_value=[])
        mock_hybrid.return_value = hybrid_instance

        await service.search(kb_id=uuid.uuid4(), query="test", include_parents=False)

    mock_expand.assert_not_called()


# ── INFO 日志 ──


async def test_info_log_contains_required_fields(caplog):
    """INFO log must include query[:80], kb_id, hit count, latency."""
    kb = _make_kb(status="active")
    kb_id = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    chunk = _make_chunk(uuid.uuid4(), kb_id=kb_id)
    # Craft query where the tail is unique: 80 'a's followed by 'TAIL_SENTINEL'
    long_query = "a" * 80 + "TAIL_SENTINEL"

    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=kb)

    from baize.knowledge.service import RetrievalService

    service = RetrievalService(
        db_session=db_session,
        qdrant_client=AsyncMock(),
        embedder_factory=MagicMock(return_value=MagicMock()),
    )

    with (
        patch("baize.knowledge.service.HybridRetriever") as mock_hybrid,
        caplog.at_level(logging.INFO, logger="baize.knowledge.service"),
    ):
        hybrid_instance = AsyncMock()
        hybrid_instance.search = AsyncMock(return_value=[chunk])
        mock_hybrid.return_value = hybrid_instance

        await service.search(kb_id=kb_id, query=long_query, include_parents=False)

    assert any(r.levelno == logging.INFO for r in caplog.records)
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records, "Expected at least one INFO log record"
    combined = " ".join(r.getMessage() for r in info_records)

    # first 80 chars present
    assert long_query[:80] in combined
    # unique tail must NOT appear (truncated)
    assert "TAIL_SENTINEL" not in combined

    # kb_id string present
    assert str(kb_id) in combined

    # hit count (1 chunk)
    assert "1" in combined


# ── tracer=None 不报错 ──


async def test_tracer_none_does_not_raise():
    """tracer=None must be accepted without any error."""
    kb = _make_kb(status="active")
    db_session = AsyncMock()
    db_session.get = AsyncMock(return_value=kb)

    from baize.knowledge.service import RetrievalService

    service = RetrievalService(
        db_session=db_session,
        qdrant_client=AsyncMock(),
        embedder_factory=MagicMock(return_value=MagicMock()),
    )

    with patch("baize.knowledge.service.HybridRetriever") as mock_hybrid:
        hybrid_instance = AsyncMock()
        hybrid_instance.search = AsyncMock(return_value=[])
        mock_hybrid.return_value = hybrid_instance

        result = await service.search(kb_id=uuid.uuid4(), query="test", tracer=None)

    assert result == []
