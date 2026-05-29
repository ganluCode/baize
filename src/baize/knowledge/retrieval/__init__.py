"""Knowledge retrieval sub-package."""

from baize.knowledge.retrieval.exceptions import (
    KnowledgeBaseNotActiveError,
    VectorDimMismatchError,
    VectorRetrievalError,
)
from baize.knowledge.retrieval.expansion import expand_parents
from baize.knowledge.retrieval.types import ScoredChunk

__all__ = [
    "ScoredChunk",
    "VectorRetrievalError",
    "KnowledgeBaseNotActiveError",
    "VectorDimMismatchError",
    "expand_parents",
]
