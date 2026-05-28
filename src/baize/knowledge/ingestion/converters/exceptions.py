"""Converter-specific exception classes."""


class UnsupportedFormatError(ValueError):
    """Raised when source_type is not handled by any registered converter."""


class DocumentDecodeError(ValueError):
    """Raised when a document cannot be decoded (e.g. wrong encoding, encrypted)."""


class ScannedPdfError(Exception):
    """Raised when a PDF contains no extractable text and likely requires OCR."""
