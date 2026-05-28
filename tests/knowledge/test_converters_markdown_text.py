"""Unit tests for markdown.py and text.py converters (F-003 acceptance criteria)."""

import pytest

from baize.knowledge.ingestion.converters import ConvertedDocument, DocumentDecodeError


class TestMarkdownConverter:
    async def test_utf8_bytes_returned_as_markdown(self):
        from baize.knowledge.ingestion.converters.markdown import convert

        content = "# Hello\n\nThis is a test."
        result = await convert(content.encode("utf-8"))
        assert isinstance(result, ConvertedDocument)
        assert result.markdown == content

    async def test_non_utf8_raises_document_decode_error(self):
        from baize.knowledge.ingestion.converters.markdown import convert

        with pytest.raises(DocumentDecodeError):
            await convert(bytes([0xFF, 0xFE, 0x00]))

    async def test_warnings_is_empty_list(self):
        from baize.knowledge.ingestion.converters.markdown import convert

        result = await convert(b"# Hello")
        assert result.warnings == []

    async def test_empty_bytes_returns_empty_markdown(self):
        from baize.knowledge.ingestion.converters.markdown import convert

        result = await convert(b"")
        assert result.markdown == ""


class TestTextConverter:
    async def test_first_line_of_each_paragraph_prefixed_with_4_spaces(self):
        from baize.knowledge.ingestion.converters.text import convert

        content = "First paragraph line1\nline2\n\nSecond paragraph\nmore"
        result = await convert(content.encode("utf-8"))
        paragraphs = result.markdown.split("\n\n")
        assert paragraphs[0].startswith("    ")
        assert paragraphs[1].startswith("    ")

    async def test_non_first_lines_in_paragraph_unchanged(self):
        from baize.knowledge.ingestion.converters.text import convert

        content = "First line\nSecond line"
        result = await convert(content.encode("utf-8"))
        lines = result.markdown.split("\n")
        assert lines[0] == "    First line"
        assert lines[1] == "Second line"

    async def test_no_lines_starting_with_hash(self):
        from baize.knowledge.ingestion.converters.text import convert

        content = "# Heading one\n\n## Heading two\n\nNormal text"
        result = await convert(content.encode("utf-8"))
        for line in result.markdown.split("\n"):
            assert not line.startswith("#"), f"Line starts with #: {line!r}"

    async def test_warnings_is_empty_list(self):
        from baize.knowledge.ingestion.converters.text import convert

        result = await convert(b"Some text")
        assert result.warnings == []

    async def test_empty_text_returns_empty_markdown(self):
        from baize.knowledge.ingestion.converters.text import convert

        result = await convert(b"")
        assert result.markdown == ""

    async def test_single_paragraph_prefixed(self):
        from baize.knowledge.ingestion.converters.text import convert

        result = await convert(b"Hello world")
        assert result.markdown == "    Hello world"
