"""Markdown pass-through converter."""

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.converters.exceptions import DocumentDecodeError


async def convert(raw_bytes: bytes) -> ConvertedDocument:
    """Decode raw bytes as UTF-8 and return as a ConvertedDocument.

    Args:
        raw_bytes: Raw bytes of a Markdown document.

    Returns:
        ConvertedDocument with the decoded markdown string and empty warnings.

    Raises:
        DocumentDecodeError: If the bytes cannot be decoded as UTF-8.
    """
    try:
        markdown = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DocumentDecodeError(f"Cannot decode document as UTF-8: {e}") from e
    return ConvertedDocument(markdown=markdown)
