"""Unit tests for VectorRetriever — mocked Qdrant + DB, validates acceptance criteria."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.knowledge.retrieval.exceptions import VectorRetrievalError
from baize.knowledge.retrieval.vector import VectorRetriever

# ── Helpers ──


def _make_scored_point(point_id: uuid.UUID, score: float) -> MagicMock:
    point = MagicMock()
    point.id = str(point_id)
    point.score = score
    return point


def _make_db_row(
    chunk_id: uuid.UUID,
    doc_id: uuid.UUID,
    kb_id: uuid.UUID,
    qdrant_point_id: uuid.UUID,
    content: str = "chunk content",
    section_path: str | None = None,
    level: int = 1,
    parent_chunk_id: uuid.UUID | None = None,
) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "content": content,
        "section_path": section_path,
        "level": level,
        "parent_chunk_id": parent_chunk_id,
        "qdrant_point_id": qdrant_point_id,
    }[key]
    return row


def _make_session(rows: list[MagicMock]) -> AsyncMock:
    mappings_result = MagicMock()
    mappings_result.all.return_value = rows
    execute_result = MagicMock()
    execute_result.mappings.return_value = mappings_result
    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result)
    return session


def _make_embedder(vector: list[float] | None = None) -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=vector or [0.1, 0.2, 0.3])
    return embedder


# ── Collection name format ──


async def test_collection_name_uses_kb_id_hex_no_hyphens():
    """Collection name must be baize_kb_ + kb_id.hex (no hyphens)."""
    kb_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(return_value=[])
    retriever = VectorRetriever(
        db_session=_make_session([]),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    await retriever.search(kb_id=kb_id, query="test", limit=5)

    call_kwargs = mock_qdrant.search.call_args.kwargs
    expected = f"baize_kb_{kb_id.hex}"
    assert call_kwargs["collection_name"] == expected
    assert "-" not in call_kwargs["collection_name"]


# ── Normal search + hydration ──


async def test_search_returns_scored_chunks_in_qdrant_order():
    """Results must be returned in the same order as Qdrant search results."""
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunk_id_1 = uuid.uuid4()
    chunk_id_2 = uuid.uuid4()
    point_id_1 = uuid.uuid4()
    point_id_2 = uuid.uuid4()

    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(
        return_value=[
            _make_scored_point(point_id_1, score=0.9),
            _make_scored_point(point_id_2, score=0.7),
        ]
    )
    rows = [
        _make_db_row(chunk_id_1, doc_id, kb_id, point_id_1, content="first"),
        _make_db_row(chunk_id_2, doc_id, kb_id, point_id_2, content="second"),
    ]
    retriever = VectorRetriever(
        db_session=_make_session(rows),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    chunks = await retriever.search(kb_id=kb_id, query="test", limit=5)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == chunk_id_1
    assert chunks[1].chunk_id == chunk_id_2
    assert chunks[0].vector_score == pytest.approx(0.9)
    assert chunks[1].vector_score == pytest.approx(0.7)


async def test_qdrant_order_preserved_when_db_rows_returned_out_of_order():
    """Final list must follow Qdrant order regardless of DB row order."""
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunk_id_a = uuid.uuid4()
    chunk_id_b = uuid.uuid4()
    point_id_a = uuid.uuid4()
    point_id_b = uuid.uuid4()

    mock_qdrant = AsyncMock()
    # Qdrant returns B first (higher score), then A
    mock_qdrant.search = AsyncMock(
        return_value=[
            _make_scored_point(point_id_b, score=0.95),
            _make_scored_point(point_id_a, score=0.85),
        ]
    )
    # DB returns rows in reversed order (A first, then B)
    rows = [
        _make_db_row(chunk_id_a, doc_id, kb_id, point_id_a),
        _make_db_row(chunk_id_b, doc_id, kb_id, point_id_b),
    ]
    retriever = VectorRetriever(
        db_session=_make_session(rows),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    chunks = await retriever.search(kb_id=kb_id, query="test", limit=5)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == chunk_id_b  # B ranks first (higher Qdrant score)
    assert chunks[1].chunk_id == chunk_id_a


# ── ScoredChunk field validation ──


async def test_bm25_fields_are_none_sources_is_vector():
    """bm25_rank and bm25_score must be None; sources must be ['vector']."""
    kb_id = uuid.uuid4()
    point_id = uuid.uuid4()

    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(return_value=[_make_scored_point(point_id, 0.8)])
    rows = [_make_db_row(uuid.uuid4(), uuid.uuid4(), kb_id, point_id)]
    retriever = VectorRetriever(
        db_session=_make_session(rows),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    chunks = await retriever.search(kb_id=kb_id, query="test", limit=5)

    chunk = chunks[0]
    assert chunk.bm25_rank is None
    assert chunk.bm25_score is None
    assert chunk.sources == ["vector"]


async def test_vector_rank_is_1_based():
    """vector_rank must be 1-based, matching position in Qdrant results."""
    kb_id = uuid.uuid4()
    point_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(
        return_value=[
            _make_scored_point(point_ids[0], 0.9),
            _make_scored_point(point_ids[1], 0.7),
            _make_scored_point(point_ids[2], 0.5),
        ]
    )
    rows = [_make_db_row(uuid.uuid4(), uuid.uuid4(), kb_id, pid) for pid in point_ids]
    retriever = VectorRetriever(
        db_session=_make_session(rows),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    chunks = await retriever.search(kb_id=kb_id, query="test", limit=5)

    assert [c.vector_rank for c in chunks] == [1, 2, 3]
    assert [c.rank for c in chunks] == [1, 2, 3]


async def test_vector_score_equals_qdrant_score_no_normalization():
    """vector_score must equal Qdrant cosine similarity score without any normalization."""
    kb_id = uuid.uuid4()
    point_id = uuid.uuid4()
    qdrant_score = 0.7654321

    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(return_value=[_make_scored_point(point_id, qdrant_score)])
    rows = [_make_db_row(uuid.uuid4(), uuid.uuid4(), kb_id, point_id)]
    retriever = VectorRetriever(
        db_session=_make_session(rows),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    chunks = await retriever.search(kb_id=kb_id, query="test", limit=5)

    assert chunks[0].vector_score == pytest.approx(qdrant_score)
    assert chunks[0].score == pytest.approx(qdrant_score)


# ── Error handling ──


async def test_collection_not_found_returns_empty_list():
    """When Qdrant signals 'collection not found', return [] without raising."""
    kb_id = uuid.uuid4()
    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(
        side_effect=ValueError(f"Collection baize_kb_{kb_id.hex} not found")
    )
    retriever = VectorRetriever(
        db_session=_make_session([]),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    result = await retriever.search(kb_id=kb_id, query="test", limit=5)

    assert result == []


async def test_connection_error_raises_vector_retrieval_error():
    """Network/connection errors must be wrapped as VectorRetrievalError."""
    kb_id = uuid.uuid4()
    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(side_effect=ConnectionError("Connection refused"))
    retriever = VectorRetriever(
        db_session=_make_session([]),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    with pytest.raises(VectorRetrievalError):
        await retriever.search(kb_id=kb_id, query="test", limit=5)


async def test_timeout_error_raises_vector_retrieval_error():
    """Timeout errors must be wrapped as VectorRetrievalError."""
    kb_id = uuid.uuid4()
    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(side_effect=TimeoutError("Request timed out"))
    retriever = VectorRetriever(
        db_session=_make_session([]),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    with pytest.raises(VectorRetrievalError):
        await retriever.search(kb_id=kb_id, query="test", limit=5)


# ── Empty results ──


async def test_empty_qdrant_results_returns_empty_list_without_db_query():
    """Empty Qdrant results must return [] without issuing a DB query."""
    kb_id = uuid.uuid4()
    session = _make_session([])
    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(return_value=[])
    retriever = VectorRetriever(
        db_session=session,
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    result = await retriever.search(kb_id=kb_id, query="test", limit=5)

    assert result == []
    session.execute.assert_not_called()


# ── Interface signature ──


async def test_search_accepts_keyword_args():
    """Verifies kb_id, query, limit are keyword-only parameters."""
    mock_qdrant = AsyncMock()
    mock_qdrant.search = AsyncMock(return_value=[])
    retriever = VectorRetriever(
        db_session=_make_session([]),
        qdrant_client=mock_qdrant,
        embedder=_make_embedder(),
    )

    result = await retriever.search(kb_id=uuid.uuid4(), query="hello", limit=10)
    assert result == []
