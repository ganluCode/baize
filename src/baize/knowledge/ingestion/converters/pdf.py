"""PDF to Markdown converter using pypdf."""

import io
import logging

from pypdf import PdfReader

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from baize.knowledge.ingestion.converters.exceptions import (
    DocumentDecodeError,
    ScannedPdfError,
)

logger = logging.getLogger(__name__)


def _flatten_outline(outline: list, level: int = 1) -> list[tuple[str, int, object]]:
    """Recursively flatten PDF outline into (title, level, destination) tuples.

    In pypdf, a nested list following a Destination represents that Destination's children.
    """
    entries = []
    for item in outline:
        if isinstance(item, list):
            entries.extend(_flatten_outline(item, level + 1))
        else:
            entries.append((item.title, level, item))
    return entries


async def convert(raw_bytes: bytes) -> ConvertedDocument:
    """Convert a PDF document to Markdown using pypdf.

    - Encrypted PDFs raise DocumentDecodeError.
    - PDFs with no extractable text raise ScannedPdfError.
    - PDFs with an outline produce heading-structured Markdown (h1/h2/h3).
    - PDFs without an outline produce plain text with a warning.
    - All results include a warning with page count and character count.

    Args:
        raw_bytes: Raw bytes of a PDF document.

    Returns:
        ConvertedDocument with markdown text and warnings.

    Raises:
        DocumentDecodeError: If the PDF is encrypted.
        ScannedPdfError: If the PDF contains no extractable text (likely scanned).
    """
    reader = PdfReader(io.BytesIO(raw_bytes))

    if reader.is_encrypted:
        raise DocumentDecodeError("无法解析加密 PDF，请先解密后重试")

    page_count = len(reader.pages)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(page_texts)

    if not full_text.strip():
        raise ScannedPdfError(
            "PDF 全文提取结果为空，可能是扫描版 PDF，请使用 OCR 工具处理后再导入"
        )

    char_count = len(full_text.strip())
    warnings: list[str] = [f"共 {page_count} 页，提取字符数 {char_count}"]

    outline = reader.outline
    if not outline:
        warnings.append("该 PDF 无 outline（书签），已退化为纯文本输出")
        return ConvertedDocument(markdown=full_text, warnings=warnings)

    entries = _flatten_outline(outline)

    outline_with_pages: list[tuple[str, int, int]] = []
    for title, level, dest in entries:
        try:
            page_num = reader.get_destination_page_number(dest)
            outline_with_pages.append((title, level, page_num))
        except Exception:
            logger.debug("Could not resolve page number for outline entry: %s", title)

    parts: list[str] = []
    for i, (title, level, page_num) in enumerate(outline_with_pages):
        heading_prefix = "#" * min(level, 3)
        parts.append(f"{heading_prefix} {title}")

        next_page = (
            outline_with_pages[i + 1][2]
            if i + 1 < len(outline_with_pages)
            else page_count
        )
        section_text = "\n".join(page_texts[page_num:next_page]).strip()
        if section_text:
            parts.append(section_text)

    markdown = "\n\n".join(parts)
    return ConvertedDocument(markdown=markdown, warnings=warnings)
