"""Unit tests for HybridRetriever — mocked BM25 + Vector, validates RRF fusion."""

import uuid
from unittest.mock import AsyncMock

import pytest

from baize.knowledge.retrieval.exceptions import VectorRetrievalError
from baize.knowledge.retrieval.types import ScoredChunk


def _make_chunk(
    chunk_id: uuid.UUID,
    *,
    rank: int = 1,
    score: float = 0.5,
    sources: list[str] | None = None,
    bm25_rank: int | None = None,
    bm25_score: float | None = None,
    vector_rank: int | None = None,
    vector_score: float | None = None,
    kb_id: uuid.UUID | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id=uuid.uuid4(),
        kb_id=kb_id or uuid.uuid4(),
        content=f"content-{chunk_id}",
        section_path=[],
        level=0,
        parent_chunk_id=None,
        score=score,
        rank=rank,
        sources=sources or ["bm25"],
        bm25_rank=bm25_rank,
        vector_rank=vector_rank,
        bm25_score=bm25_score,
        vector_score=vector_score,
    )


_RRF_K = 60


# ── 接口签名 ──


async def test_search_accepts_keyword_args():
    """kb_id, query, top_k, candidate_limit must be keyword-only."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=[])
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    result = await retriever.search(kb_id=uuid.uuid4(), query="test", top_k=5, candidate_limit=20)
    assert result == []


# ── 两路并发 ──


async def test_both_retrievers_called_concurrently():
    """BM25 and Vector search must be invoked (via asyncio.gather, not serially)."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=[])
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    await retriever.search(kb_id=kb_id, query="hello", top_k=5, candidate_limit=20)

    bm25.search.assert_called_once_with(kb_id=kb_id, query="hello", limit=20)
    vector.search.assert_called_once_with(kb_id=kb_id, query="hello", limit=20)


# ── RRF 分数计算：两路交集 ──


async def test_rrf_score_for_overlapping_chunks():
    """Chunks appearing in both BM25 and Vector must have RRF = 1/(k+bm25_rank) + 1/(k+vector_rank)."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    shared_id = uuid.uuid4()

    bm25_chunk = _make_chunk(shared_id, rank=1, score=0.9, sources=["bm25"], bm25_rank=1, bm25_score=0.9, kb_id=kb_id)
    vector_chunk = _make_chunk(
        shared_id, rank=2, score=0.8, sources=["vector"], vector_rank=2, vector_score=0.8, kb_id=kb_id
    )

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=[bm25_chunk])
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[vector_chunk])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=kb_id, query="test", top_k=10, candidate_limit=20)

    assert len(results) == 1
    expected_rrf = 1.0 / (_RRF_K + 1) + 1.0 / (_RRF_K + 2)
    assert results[0].score == pytest.approx(expected_rrf, abs=1e-10)
    assert results[0].bm25_rank == 1
    assert results[0].vector_rank == 2
    assert results[0].bm25_score == pytest.approx(0.9)
    assert results[0].vector_score == pytest.approx(0.8)
    assert sorted(results[0].sources) == ["bm25", "vector"]


# ── RRF 分数计算：仅 BM25 命中 ──


async def test_rrf_score_bm25_only_chunk():
    """Chunk only in BM25 has RRF = 1/(k+bm25_rank), vector_rank must be None."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    bm25_chunk = _make_chunk(
        chunk_id, rank=3, score=0.7, sources=["bm25"], bm25_rank=3, bm25_score=0.7, kb_id=kb_id
    )

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=[bm25_chunk])
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=kb_id, query="test", top_k=10, candidate_limit=20)

    assert len(results) == 1
    expected_rrf = 1.0 / (_RRF_K + 3)
    assert results[0].score == pytest.approx(expected_rrf, abs=1e-10)
    assert results[0].bm25_rank == 3
    assert results[0].vector_rank is None
    assert results[0].sources == ["bm25"]


# ── RRF 分数计算：仅 Vector 命中 ──


async def test_rrf_score_vector_only_chunk():
    """Chunk only in Vector has RRF = 1/(k+vector_rank), bm25_rank must be None."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    vector_chunk = _make_chunk(
        chunk_id, rank=1, score=0.95, sources=["vector"], vector_rank=1, vector_score=0.95, kb_id=kb_id
    )

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=[])
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[vector_chunk])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=kb_id, query="test", top_k=10, candidate_limit=20)

    assert len(results) == 1
    expected_rrf = 1.0 / (_RRF_K + 1)
    assert results[0].score == pytest.approx(expected_rrf, abs=1e-10)
    assert results[0].bm25_rank is None
    assert results[0].vector_rank == 1
    assert results[0].sources == ["vector"]


# ── 排序 + top_k 截断 ──


async def test_results_sorted_by_rrf_desc_and_top_k_applied():
    """Results must be sorted by RRF score descending; top_k limits output."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    ids = [uuid.uuid4() for _ in range(5)]

    bm25_chunks = [
        _make_chunk(ids[0], rank=1, bm25_rank=1, bm25_score=0.9, sources=["bm25"], kb_id=kb_id),
        _make_chunk(ids[1], rank=2, bm25_rank=2, bm25_score=0.7, sources=["bm25"], kb_id=kb_id),
        _make_chunk(ids[2], rank=3, bm25_rank=3, bm25_score=0.5, sources=["bm25"], kb_id=kb_id),
    ]
    vector_chunks = [
        _make_chunk(ids[0], rank=1, vector_rank=1, vector_score=0.95, sources=["vector"], kb_id=kb_id),
        _make_chunk(ids[3], rank=2, vector_rank=2, vector_score=0.85, sources=["vector"], kb_id=kb_id),
        _make_chunk(ids[4], rank=3, vector_rank=3, vector_score=0.75, sources=["vector"], kb_id=kb_id),
    ]

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=bm25_chunks)
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=vector_chunks)
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=kb_id, query="test", top_k=3, candidate_limit=20)

    assert len(results) == 3
    # ids[0] hit both paths → highest RRF
    assert results[0].chunk_id == ids[0]
    # Verify descending order
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


# ── rank 字段 1-based ──


async def test_rank_field_is_1_based_after_rrf():
    """After RRF sorting, rank must be 1-based sequential."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    ids = [uuid.uuid4() for _ in range(3)]

    bm25_chunks = [
        _make_chunk(ids[i], rank=i + 1, bm25_rank=i + 1, bm25_score=0.9 - i * 0.1, sources=["bm25"], kb_id=kb_id)
        for i in range(3)
    ]

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=bm25_chunks)
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=kb_id, query="test", top_k=10, candidate_limit=20)

    assert [c.rank for c in results] == [1, 2, 3]


# ── sources 字段 ──


async def test_sources_field_reflects_hit_origins():
    """sources must be ['bm25'], ['vector'], or ['bm25', 'vector'] based on which paths hit."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    both_id = uuid.uuid4()
    bm25_only_id = uuid.uuid4()
    vector_only_id = uuid.uuid4()

    bm25_chunks = [
        _make_chunk(both_id, rank=1, bm25_rank=1, bm25_score=0.9, sources=["bm25"], kb_id=kb_id),
        _make_chunk(bm25_only_id, rank=2, bm25_rank=2, bm25_score=0.7, sources=["bm25"], kb_id=kb_id),
    ]
    vector_chunks = [
        _make_chunk(both_id, rank=1, vector_rank=1, vector_score=0.8, sources=["vector"], kb_id=kb_id),
        _make_chunk(vector_only_id, rank=2, vector_rank=2, vector_score=0.6, sources=["vector"], kb_id=kb_id),
    ]

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=bm25_chunks)
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=vector_chunks)
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=kb_id, query="test", top_k=10, candidate_limit=20)

    result_map = {r.chunk_id: r for r in results}
    assert sorted(result_map[both_id].sources) == ["bm25", "vector"]
    assert result_map[bm25_only_id].sources == ["bm25"]
    assert result_map[vector_only_id].sources == ["vector"]


# ── 两路均空 ──


async def test_both_empty_returns_empty_list():
    """When both retrievers return [], search returns []."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=[])
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=uuid.uuid4(), query="nothing", top_k=5, candidate_limit=20)
    assert results == []


# ── BM25 非连接类异常 → 降级为空 + warning ──


async def test_bm25_non_connection_error_degrades_to_empty_with_warning(caplog):
    """Non-connection exception from BM25 must be caught, logged as warning, treated as empty."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    vector_chunk = _make_chunk(
        chunk_id, rank=1, vector_rank=1, vector_score=0.9, sources=["vector"], kb_id=kb_id
    )

    bm25 = AsyncMock()
    bm25.search = AsyncMock(side_effect=RuntimeError("some DB error"))
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=[vector_chunk])
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    import logging

    with caplog.at_level(logging.WARNING):
        results = await retriever.search(kb_id=kb_id, query="test", top_k=10, candidate_limit=20)

    assert len(results) == 1
    assert results[0].chunk_id == chunk_id
    assert any("bm25" in r.message.lower() for r in caplog.records if r.levelno == logging.WARNING)


# ── Vector 抛 VectorRetrievalError → 向上传播 ──


async def test_vector_retrieval_error_propagates():
    """VectorRetrievalError from vector path must propagate, not be swallowed."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=[])
    vector = AsyncMock()
    vector.search = AsyncMock(side_effect=VectorRetrievalError("Qdrant connection refused"))
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    with pytest.raises(VectorRetrievalError):
        await retriever.search(kb_id=uuid.uuid4(), query="test", top_k=5, candidate_limit=20)


# ── 去重合并：5+5 有 2 条重复 → 8 条候选 ──


async def test_dedup_merge_5_plus_5_with_2_overlap():
    """5 BM25 + 5 Vector results with 2 shared chunk_ids produce 8 unique candidates."""
    from baize.knowledge.retrieval.hybrid import HybridRetriever

    kb_id = uuid.uuid4()
    shared_ids = [uuid.uuid4(), uuid.uuid4()]
    bm25_only_ids = [uuid.uuid4() for _ in range(3)]
    vector_only_ids = [uuid.uuid4() for _ in range(3)]

    bm25_chunks = []
    for i, cid in enumerate(shared_ids + bm25_only_ids):
        bm25_chunks.append(
            _make_chunk(cid, rank=i + 1, bm25_rank=i + 1, bm25_score=0.9 - i * 0.1, sources=["bm25"], kb_id=kb_id)
        )

    vector_chunks = []
    for i, cid in enumerate(shared_ids + vector_only_ids):
        vector_chunks.append(
            _make_chunk(
                cid, rank=i + 1, vector_rank=i + 1, vector_score=0.95 - i * 0.1, sources=["vector"], kb_id=kb_id
            )
        )

    bm25 = AsyncMock()
    bm25.search = AsyncMock(return_value=bm25_chunks)
    vector = AsyncMock()
    vector.search = AsyncMock(return_value=vector_chunks)
    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    results = await retriever.search(kb_id=kb_id, query="test", top_k=8, candidate_limit=20)

    assert len(results) == 8
    result_ids = {r.chunk_id for r in results}
    assert result_ids == set(shared_ids + bm25_only_ids + vector_only_ids)

    for r in results:
        if r.chunk_id in shared_ids:
            assert sorted(r.sources) == ["bm25", "vector"]
            expected = 1.0 / (_RRF_K + r.bm25_rank) + 1.0 / (_RRF_K + r.vector_rank)
            assert r.score == pytest.approx(expected, abs=1e-10)
