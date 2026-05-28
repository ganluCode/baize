"""DOCX to Markdown converter using mammoth."""

import io

import mammoth

from baize.knowledge.ingestion.converters._types import ConvertedDocument


def _discard_image(image) -> list:
    """Image hook that discards all images by returning an empty node list."""
    return []


async def convert(raw_bytes: bytes) -> ConvertedDocument:
    """Convert a DOCX document to Markdown using mammoth.

    Images are discarded via convert_image hook (no base64 or img tags in output).
    Mammoth warning messages are collected into ConvertedDocument.warnings.

    Args:
        raw_bytes: Raw bytes of a .docx document.

    Returns:
        ConvertedDocument with markdown text and any mammoth warnings.
    """
    result = mammoth.convert_to_markdown(io.BytesIO(raw_bytes), convert_image=_discard_image)

    warnings = [
        msg.message
        for msg in result.messages
        if msg.type == "warning"
    ]

    return ConvertedDocument(markdown=result.value, warnings=warnings)
