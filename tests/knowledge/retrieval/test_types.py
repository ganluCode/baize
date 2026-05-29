"""Tests for retrieval module types and exceptions."""

import dataclasses
import uuid

import pytest

from baize.knowledge.retrieval import ScoredChunk
from baize.knowledge.retrieval.exceptions import (
    KnowledgeBaseNotActiveError,
    VectorDimMismatchError,
    VectorRetrievalError,
)
from baize.knowledge.retrieval.types import ScoredChunk as ScoredChunkDirect


def test_scored_chunk_importable_from_both_paths():
    assert ScoredChunk is ScoredChunkDirect


def test_scored_chunk_has_all_fields():
    field_names = {f.name for f in dataclasses.fields(ScoredChunk)}
    expected = {
        "chunk_id",
        "doc_id",
        "kb_id",
        "content",
        "section_path",
        "level",
        "parent_chunk_id",
        "score",
        "rank",
        "sources",
        "bm25_rank",
        "vector_rank",
        "bm25_score",
        "vector_score",
    }
    assert field_names == expected


def test_exception_hierarchy():
    assert issubclass(VectorRetrievalError, Exception)
    assert issubclass(KnowledgeBaseNotActiveError, Exception)
    assert issubclass(VectorDimMismatchError, Exception)


def test_exceptions_can_be_caught_as_base_exception():
    with pytest.raises(Exception):
        raise VectorRetrievalError("qdrant unreachable")

    with pytest.raises(Exception):
        raise KnowledgeBaseNotActiveError("kb deleted")

    with pytest.raises(Exception):
        raise VectorDimMismatchError("dim=768 expected 1536")
