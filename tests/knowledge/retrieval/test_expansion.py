"""Unit tests for expand_parents — mocked DB, validates acceptance criteria."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.knowledge.retrieval.expansion import expand_parents
from baize.knowledge.retrieval.types import ScoredChunk


def _make_scored_chunk(
    chunk_id: uuid.UUID | None = None,
    doc_id: uuid.UUID | None = None,
    kb_id: uuid.UUID | None = None,
    content: str = "content",
    level: int = 1,
    parent_chunk_id: uuid.UUID | None = None,
    score: float = 0.5,
    rank: int = 1,
    sources: list[str] | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        doc_id=doc_id or uuid.uuid4(),
        kb_id=kb_id or uuid.uuid4(),
        content=content,
        section_path=[],
        level=level,
        parent_chunk_id=parent_chunk_id,
        score=score,
        rank=rank,
        sources=sources or ["bm25"],
        bm25_rank=rank,
        vector_rank=None,
        bm25_score=score,
        vector_score=None,
    )


def _make_db_row(
    chunk_id: uuid.UUID,
    doc_id: uuid.UUID,
    kb_id: uuid.UUID,
    content: str = "parent content",
    level: int = 0,
    parent_chunk_id: uuid.UUID | None = None,
    section_path: list[str] | None = None,
) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "content": content,
        "level": level,
        "parent_chunk_id": parent_chunk_id,
        "section_path": json.dumps(section_path) if section_path else None,
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


# ── 基本扩展：2 个 child → 2 个 parent 追加在末尾 ──


async def test_two_children_expand_to_two_parents_appended():
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    parent_id_1 = uuid.uuid4()
    parent_id_2 = uuid.uuid4()
    child_1 = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=parent_id_1, rank=1)
    child_2 = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=parent_id_2, rank=2)

    parent_row_1 = _make_db_row(parent_id_1, doc_id, kb_id, content="parent 1", level=0)
    parent_row_2 = _make_db_row(parent_id_2, doc_id, kb_id, content="parent 2", level=0)
    session = _make_session([parent_row_1, parent_row_2])

    result = await expand_parents(db_session=session, chunks=[child_1, child_2])

    assert len(result) == 4
    # original order preserved
    assert result[0] is child_1
    assert result[1] is child_2
    # parents appended
    parent_ids_in_result = {c.chunk_id for c in result[2:]}
    assert parent_id_1 in parent_ids_in_result
    assert parent_id_2 in parent_ids_in_result


# ── parent 字段：score=0.0, sources=["parent_expansion"] ──


async def test_parent_chunk_score_and_sources():
    kb_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    child = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=parent_id, rank=1)

    parent_row = _make_db_row(parent_id, uuid.uuid4(), kb_id, level=0)
    session = _make_session([parent_row])

    result = await expand_parents(db_session=session, chunks=[child])

    parent = result[1]
    assert parent.chunk_id == parent_id
    assert parent.score == 0.0
    assert parent.sources == ["parent_expansion"]
    assert parent.bm25_rank is None
    assert parent.vector_rank is None
    assert parent.bm25_score is None
    assert parent.vector_score is None


# ── rank 从 len(原列表)+1 开始 ──


async def test_parent_rank_starts_after_original():
    kb_id = uuid.uuid4()
    parent_id_1 = uuid.uuid4()
    parent_id_2 = uuid.uuid4()
    child_1 = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=parent_id_1, rank=1)
    child_2 = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=parent_id_2, rank=2)

    # rows returned in the order parent_id_1, parent_id_2
    row_1 = _make_db_row(parent_id_1, uuid.uuid4(), kb_id, level=0)
    row_2 = _make_db_row(parent_id_2, uuid.uuid4(), kb_id, level=0)
    session = _make_session([row_1, row_2])

    result = await expand_parents(db_session=session, chunks=[child_1, child_2])

    parents = result[2:]
    ranks = {c.chunk_id: c.rank for c in parents}
    # rank values are len(original)+1=3 and len(original)+2=4, order by parent_id_1 then parent_id_2
    assert ranks[parent_id_1] == 3
    assert ranks[parent_id_2] == 4


# ── 去重：parent 已在原列表中则不重复 ──


async def test_no_duplicate_if_parent_already_in_original():
    kb_id = uuid.uuid4()
    # parent_id_1 is also a chunk in the original list
    parent_id_1 = uuid.uuid4()
    existing_parent = _make_scored_chunk(chunk_id=parent_id_1, kb_id=kb_id, level=0, parent_chunk_id=None, rank=1)
    child = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=parent_id_1, rank=2)

    session = _make_session([])

    result = await expand_parents(db_session=session, chunks=[existing_parent, child])

    # No new chunks appended because parent is already present
    assert len(result) == 2
    assert result[0] is existing_parent
    assert result[1] is child
    # No DB call needed since the only parent is already in the list
    session.execute.assert_not_called()


# ── level=0 或 parent_chunk_id=None 的 chunk 被跳过 ──


async def test_level_zero_chunks_skipped_no_db_query():
    chunk = _make_scored_chunk(level=0, parent_chunk_id=None, rank=1)
    session = _make_session([])

    result = await expand_parents(db_session=session, chunks=[chunk])

    assert len(result) == 1
    assert result[0] is chunk
    session.execute.assert_not_called()


async def test_none_parent_chunk_id_skipped_no_db_query():
    chunk = _make_scored_chunk(level=1, parent_chunk_id=None, rank=1)
    session = _make_session([])

    result = await expand_parents(db_session=session, chunks=[chunk])

    assert len(result) == 1
    session.execute.assert_not_called()


# ── 不递归：只批量查一次，不查 parent 的 parent ──


async def test_no_recursion_db_called_at_most_once():
    kb_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    grandparent_id = uuid.uuid4()
    child = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=parent_id, rank=1)

    # parent row itself has a parent_chunk_id (grandparent)
    parent_row = _make_db_row(parent_id, uuid.uuid4(), kb_id, level=1, parent_chunk_id=grandparent_id)
    session = _make_session([parent_row])

    result = await expand_parents(db_session=session, chunks=[child])

    # DB called exactly once (batch fetch), no recursive second call
    assert session.execute.call_count == 1
    # grandparent NOT in result
    result_ids = {c.chunk_id for c in result}
    assert grandparent_id not in result_ids


# ── 空输入 ──


async def test_empty_chunks_returns_empty_no_db_call():
    session = _make_session([])

    result = await expand_parents(db_session=session, chunks=[])

    assert result == []
    session.execute.assert_not_called()


# ── 所有 chunk 均无 parent → 不发 DB 查询 ──


async def test_all_chunks_without_parent_no_db_call():
    chunks = [
        _make_scored_chunk(level=0, parent_chunk_id=None, rank=1),
        _make_scored_chunk(level=1, parent_chunk_id=None, rank=2),
    ]
    session = _make_session([])

    result = await expand_parents(db_session=session, chunks=chunks)

    assert len(result) == 2
    session.execute.assert_not_called()


# ── 去重：两个 child 指向同一 parent，只追加一次 ──


async def test_two_children_same_parent_deduped():
    kb_id = uuid.uuid4()
    shared_parent_id = uuid.uuid4()
    child_1 = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=shared_parent_id, rank=1)
    child_2 = _make_scored_chunk(kb_id=kb_id, level=1, parent_chunk_id=shared_parent_id, rank=2)

    parent_row = _make_db_row(shared_parent_id, uuid.uuid4(), kb_id, level=0)
    session = _make_session([parent_row])

    result = await expand_parents(db_session=session, chunks=[child_1, child_2])

    # Only one parent appended
    assert len(result) == 3
    assert result[2].chunk_id == shared_parent_id
    # Exactly one DB call
    session.execute.call_count == 1
