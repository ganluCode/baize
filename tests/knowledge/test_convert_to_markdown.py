"""Tests for convert_to_markdown unified entry point (F-006)."""

import pytest

from baize.knowledge.ingestion.converters import (
    ConvertedDocument,
    UnsupportedFormatError,
    convert_to_markdown,
)


@pytest.mark.asyncio
async def test_source_type_markdown():
    raw = "# Title\n\nHello world".encode("utf-8")
    doc = await convert_to_markdown(source_type="markdown", raw_bytes=raw)
    assert isinstance(doc, ConvertedDocument)
    assert "# Title" in doc.markdown


@pytest.mark.asyncio
async def test_source_type_raw_text():
    raw = "Hello world\n\nSecond paragraph".encode("utf-8")
    doc = await convert_to_markdown(source_type="raw_text", raw_bytes=raw)
    assert isinstance(doc, ConvertedDocument)
    assert "Hello world" in doc.markdown


@pytest.mark.asyncio
async def test_source_type_docx():
    import io
    import zipfile

    # Minimal valid docx is a zip with required structure
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Test content</w:t></w:r></w:p>"
            "</w:body>"
            "</w:document>",
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            "</Relationships>",
        )
    docx_bytes = buf.getvalue()

    doc = await convert_to_markdown(source_type="docx", raw_bytes=docx_bytes)
    assert isinstance(doc, ConvertedDocument)
    assert isinstance(doc.markdown, str)


@pytest.mark.asyncio
async def test_source_type_pdf():
    from unittest.mock import MagicMock, patch

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page content here"

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [mock_page]
    mock_reader.outline = []

    with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=mock_reader):
        doc = await convert_to_markdown(
            source_type="pdf",
            raw_bytes=b"%PDF-1.4 fake",
            filename="test.pdf",
        )
    assert isinstance(doc, ConvertedDocument)
    assert "Page content here" in doc.markdown


@pytest.mark.asyncio
async def test_unsupported_source_type_raises():
    with pytest.raises(UnsupportedFormatError):
        await convert_to_markdown(source_type="html", raw_bytes=b"<html></html>")


@pytest.mark.asyncio
async def test_unsupported_source_type_unknown():
    with pytest.raises(UnsupportedFormatError):
        await convert_to_markdown(source_type="xlsx", raw_bytes=b"data")


@pytest.mark.asyncio
async def test_function_signature_keyword_only():
    """convert_to_markdown should accept keyword-only arguments."""
    raw = b"# hello"
    doc = await convert_to_markdown(source_type="markdown", raw_bytes=raw, filename="test.md")
    assert isinstance(doc, ConvertedDocument)


@pytest.mark.asyncio
async def test_filename_optional():
    raw = b"some text"
    doc = await convert_to_markdown(source_type="raw_text", raw_bytes=raw)
    assert isinstance(doc, ConvertedDocument)
