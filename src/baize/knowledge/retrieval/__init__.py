"""Knowledge retrieval sub-package."""

from baize.knowledge.retrieval.exceptions import (
    KnowledgeBaseNotActiveError,
    VectorDimMismatchError,
    VectorRetrievalError,
)
from baize.knowledge.retrieval.types import ScoredChunk

__all__ = [
    "ScoredChunk",
    "VectorRetrievalError",
    "KnowledgeBaseNotActiveError",
    "VectorDimMismatchError",
]
