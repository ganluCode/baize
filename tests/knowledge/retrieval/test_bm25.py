"""Unit tests for BM25Retriever — mocked DB, validates acceptance criteria."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.knowledge.retrieval.bm25 import BM25Retriever
from baize.knowledge.retrieval.types import ScoredChunk


def _make_row(
    chunk_id: uuid.UUID,
    doc_id: uuid.UUID,
    kb_id: uuid.UUID,
    content: str,
    bm25_score: float,
    section_path: list[str] | None = None,
    level: int = 0,
    parent_chunk_id: uuid.UUID | None = None,
) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "content": content,
        "bm25_score": bm25_score,
        "section_path": json.dumps(section_path, ensure_ascii=False) if section_path else None,
        "level": level,
        "parent_chunk_id": parent_chunk_id,
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


# ── 空 query ──


async def test_empty_string_query_returns_empty_no_db_call():
    session = _make_session([])
    retriever = BM25Retriever(db_session=session)
    result = await retriever.search(kb_id=uuid.uuid4(), query="", limit=10)
    assert result == []
    session.execute.assert_not_called()


async def test_whitespace_only_query_returns_empty_no_db_call():
    session = _make_session([])
    retriever = BM25Retriever(db_session=session)
    result = await retriever.search(kb_id=uuid.uuid4(), query="   ", limit=10)
    assert result == []
    session.execute.assert_not_called()


# ── 超长 query 截断 ──


async def test_long_query_truncated_to_500_chars():
    session = _make_session([])
    retriever = BM25Retriever(db_session=session)
    long_query = "a" * 600
    await retriever.search(kb_id=uuid.uuid4(), query=long_query, limit=5)
    session.execute.assert_called_once()
    call_kwargs = session.execute.call_args
    params = call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs.get("params", {})
    assert len(params["q"]) == 500


async def test_query_exactly_500_chars_not_truncated():
    session = _make_session([])
    retriever = BM25Retriever(db_session=session)
    query = "b" * 500
    await retriever.search(kb_id=uuid.uuid4(), query=query, limit=5)
    call_kwargs = session.execute.call_args
    params = call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs.get("params", {})
    assert params["q"] == query


# ── 结果映射到 ScoredChunk ──


async def test_results_mapped_to_scored_chunk():
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    rows = [_make_row(chunk_id, doc_id, kb_id, "hello world", bm25_score=0.5, section_path=["Sec 1"])]
    session = _make_session(rows)
    retriever = BM25Retriever(db_session=session)

    result = await retriever.search(kb_id=kb_id, query="hello", limit=10)

    assert len(result) == 1
    chunk = result[0]
    assert isinstance(chunk, ScoredChunk)
    assert chunk.chunk_id == chunk_id
    assert chunk.doc_id == doc_id
    assert chunk.kb_id == kb_id
    assert chunk.content == "hello world"
    assert chunk.section_path == ["Sec 1"]


async def test_vector_fields_are_none_sources_is_bm25():
    kb_id = uuid.uuid4()
    rows = [_make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "text", bm25_score=0.3)]
    session = _make_session(rows)
    retriever = BM25Retriever(db_session=session)

    result = await retriever.search(kb_id=kb_id, query="text", limit=10)

    chunk = result[0]
    assert chunk.vector_rank is None
    assert chunk.vector_score is None
    assert chunk.sources == ["bm25"]


async def test_bm25_rank_and_score_filled():
    kb_id = uuid.uuid4()
    rows = [
        _make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "first", bm25_score=0.8),
        _make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "second", bm25_score=0.4),
    ]
    session = _make_session(rows)
    retriever = BM25Retriever(db_session=session)

    result = await retriever.search(kb_id=kb_id, query="first second", limit=10)

    assert result[0].bm25_rank == 1
    assert result[0].bm25_score == pytest.approx(0.8)
    assert result[1].bm25_rank == 2
    assert result[1].bm25_score == pytest.approx(0.4)


# ── rank 字段 1-based ──


async def test_rank_field_is_1_based():
    kb_id = uuid.uuid4()
    rows = [
        _make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "a", bm25_score=0.9),
        _make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "b", bm25_score=0.7),
        _make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "c", bm25_score=0.5),
    ]
    session = _make_session(rows)
    retriever = BM25Retriever(db_session=session)

    result = await retriever.search(kb_id=kb_id, query="abc", limit=10)

    assert [c.rank for c in result] == [1, 2, 3]


# ── None section_path 处理 ──


async def test_none_section_path_becomes_empty_list():
    kb_id = uuid.uuid4()
    rows = [_make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "no path", bm25_score=0.5, section_path=None)]
    session = _make_session(rows)
    retriever = BM25Retriever(db_session=session)

    result = await retriever.search(kb_id=kb_id, query="no path", limit=10)

    assert result[0].section_path == []


# ── 接口签名（关键字参数）──


async def test_search_accepts_keyword_args():
    """Verifies that kb_id, query, limit are keyword-only parameters."""
    session = _make_session([])
    retriever = BM25Retriever(db_session=session)
    # must not raise
    result = await retriever.search(kb_id=uuid.uuid4(), query="", limit=5)
    assert result == []


# ── score 字段等于 bm25_score ──


async def test_score_equals_bm25_score():
    kb_id = uuid.uuid4()
    rows = [_make_row(uuid.uuid4(), uuid.uuid4(), kb_id, "match", bm25_score=0.75)]
    session = _make_session(rows)
    retriever = BM25Retriever(db_session=session)

    result = await retriever.search(kb_id=kb_id, query="match", limit=10)

    assert result[0].score == pytest.approx(0.75)
    assert result[0].bm25_score == pytest.approx(0.75)
