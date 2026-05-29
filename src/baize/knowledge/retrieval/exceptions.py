"""Custom exceptions for the retrieval layer."""


class VectorRetrievalError(Exception):
    """Raised when Qdrant connection fails or an unexpected network error occurs."""


class KnowledgeBaseNotActiveError(Exception):
    """Raised when attempting to search a knowledge base whose status is not 'active'."""


class VectorDimMismatchError(Exception):
    """Raised when an embedder's output dimension differs from the collection's configured dimension."""
