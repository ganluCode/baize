"""Exceptions for the knowledge ingestion pipeline."""


class DocumentAlreadyExistsError(Exception):
    """Raised when a document with the same content hash already exists in the KB."""
