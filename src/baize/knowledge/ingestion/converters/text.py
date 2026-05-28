"""Plain text to Markdown converter."""

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.converters.exceptions import DocumentDecodeError


async def convert(raw_bytes: bytes) -> ConvertedDocument:
    """Convert plain text bytes to safe Markdown.

    Splits text on double newlines to identify paragraphs and adds a 4-space
    prefix to the first line of each paragraph. This prevents any line
    starting with '#' from being interpreted as a Markdown heading.

    Args:
        raw_bytes: Raw bytes of a plain text document.

    Returns:
        ConvertedDocument with no lines starting with '#' and empty warnings.

    Raises:
        DocumentDecodeError: If the bytes cannot be decoded as UTF-8.
    """
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DocumentDecodeError(f"Cannot decode document as UTF-8: {e}") from e

    if not text:
        return ConvertedDocument(markdown="")

    paragraphs = text.split("\n\n")
    processed: list[str] = []
    for para in paragraphs:
        if not para:
            processed.append(para)
            continue
        lines = para.split("\n")
        lines[0] = "    " + lines[0]
        processed.append("\n".join(lines))

    return ConvertedDocument(markdown="\n\n".join(processed))
