"""Document converters — convert raw bytes to Markdown.

Public API (importable directly from this package):
  - ConvertedDocument
  - UnsupportedFormatError
  - DocumentDecodeError
  - ScannedPdfError
  - convert_to_markdown
"""

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.converters.exceptions import (
    DocumentDecodeError,
    ScannedPdfError,
    UnsupportedFormatError,
)
from baize.knowledge.ingestion.converters import (
    docx as _docx,
    markdown as _markdown,
    pdf as _pdf,
    text as _text,
)

_CONVERTERS = {
    "markdown": _markdown.convert,
    "raw_text": _text.convert,
    "docx": _docx.convert,
    "pdf": _pdf.convert,
}


async def convert_to_markdown(
    *,
    source_type: str,
    raw_bytes: bytes,
    filename: str | None = None,
) -> ConvertedDocument:
    """Convert raw document bytes to Markdown using the appropriate converter.

    Args:
        source_type: Document type — one of 'markdown', 'raw_text', 'docx', 'pdf'.
        raw_bytes: Raw bytes of the document.
        filename: Optional original filename (currently unused, reserved for future use).

    Returns:
        ConvertedDocument with the markdown text and any conversion warnings.

    Raises:
        UnsupportedFormatError: If source_type is not supported.
        DocumentDecodeError: If the document cannot be decoded.
        ScannedPdfError: If the PDF contains no extractable text.
    """
    converter = _CONVERTERS.get(source_type)
    if converter is None:
        raise UnsupportedFormatError(
            f"Unsupported source_type: {source_type!r}. "
            f"Supported types: {sorted(_CONVERTERS)}"
        )
    return await converter(raw_bytes)


__all__ = [
    "ConvertedDocument",
    "DocumentDecodeError",
    "ScannedPdfError",
    "UnsupportedFormatError",
    "convert_to_markdown",
]
