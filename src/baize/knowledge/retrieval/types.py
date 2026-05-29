"""Data types for the retrieval layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class ScoredChunk:
    """A knowledge chunk augmented with retrieval scoring metadata."""

    chunk_id: uuid.UUID
    doc_id: uuid.UUID
    kb_id: uuid.UUID
    content: str
    section_path: list[str]
    level: int
    parent_chunk_id: uuid.UUID | None
    score: float
    rank: int
    sources: list[str]
    bm25_rank: int | None
    vector_rank: int | None
    bm25_score: float | None
    vector_score: float | None
