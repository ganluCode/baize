"""Document converters — convert raw bytes to Markdown.

Public API (importable directly from this package):
  - ConvertedDocument
  - UnsupportedFormatError
  - DocumentDecodeError
  - ScannedPdfError
  - convert_to_markdown  (added in F-006)
"""

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.converters.exceptions import (
    DocumentDecodeError,
    ScannedPdfError,
    UnsupportedFormatError,
)

__all__ = [
    "ConvertedDocument",
    "DocumentDecodeError",
    "ScannedPdfError",
    "UnsupportedFormatError",
]
