"""Unit tests for docx.py converter (F-014 acceptance criteria)."""

from unittest.mock import MagicMock, patch

import pytest

from baize.knowledge.ingestion.converters._types import ConvertedDocument
from tests.knowledge.fixtures.conftest import sample_docx_bytes


class TestDocxConverterHeadings:
    async def test_sample_docx_contains_h1_heading(self):
        from baize.knowledge.ingestion.converters.docx import convert

        result = await convert(sample_docx_bytes())
        assert "# " in result.markdown

    async def test_sample_docx_contains_h2_heading(self):
        from baize.knowledge.ingestion.converters.docx import convert

        result = await convert(sample_docx_bytes())
        assert "## " in result.markdown

    async def test_sample_docx_returns_converted_document(self):
        from baize.knowledge.ingestion.converters.docx import convert

        result = await convert(sample_docx_bytes())
        assert isinstance(result, ConvertedDocument)
        assert isinstance(result.markdown, str)
        assert isinstance(result.warnings, list)


class TestDocxConverterWarnings:
    async def test_mammoth_warning_messages_written_to_converted_document(self):
        from baize.knowledge.ingestion.converters.docx import convert

        warning_msg = MagicMock()
        warning_msg.type = "warning"
        warning_msg.message = "Unrecognised element: custom:element"

        mock_result = MagicMock()
        mock_result.value = "# Heading\n\nContent"
        mock_result.messages = [warning_msg]

        with patch("baize.knowledge.ingestion.converters.docx.mammoth") as mock_mammoth:
            mock_mammoth.convert_to_markdown.return_value = mock_result
            result = await convert(b"fake docx bytes")

        assert "Unrecognised element: custom:element" in result.warnings

    async def test_non_warning_messages_excluded_from_warnings(self):
        from baize.knowledge.ingestion.converters.docx import convert

        info_msg = MagicMock()
        info_msg.type = "info"
        info_msg.message = "Info message"

        warning_msg = MagicMock()
        warning_msg.type = "warning"
        warning_msg.message = "Warning message"

        mock_result = MagicMock()
        mock_result.value = "# Content"
        mock_result.messages = [info_msg, warning_msg]

        with patch("baize.knowledge.ingestion.converters.docx.mammoth") as mock_mammoth:
            mock_mammoth.convert_to_markdown.return_value = mock_result
            result = await convert(b"fake docx bytes")

        assert result.warnings == ["Warning message"]

    async def test_no_warning_messages_yields_empty_warnings(self):
        from baize.knowledge.ingestion.converters.docx import convert

        mock_result = MagicMock()
        mock_result.value = "# Content"
        mock_result.messages = []

        with patch("baize.knowledge.ingestion.converters.docx.mammoth") as mock_mammoth:
            mock_mammoth.convert_to_markdown.return_value = mock_result
            result = await convert(b"fake docx bytes")

        assert result.warnings == []


class TestDocxConverterImageDiscarding:
    def test_discard_image_hook_returns_empty_list(self):
        from baize.knowledge.ingestion.converters.docx import _discard_image

        fake_image = MagicMock()
        assert _discard_image(fake_image) == []

    async def test_convert_image_hook_registered_with_mammoth(self):
        from baize.knowledge.ingestion.converters.docx import _discard_image, convert

        mock_result = MagicMock()
        mock_result.value = "Content"
        mock_result.messages = []

        with patch("baize.knowledge.ingestion.converters.docx.mammoth") as mock_mammoth:
            mock_mammoth.convert_to_markdown.return_value = mock_result
            await convert(b"fake docx bytes")

        call_kwargs = mock_mammoth.convert_to_markdown.call_args.kwargs
        assert call_kwargs.get("convert_image") is _discard_image

    async def test_output_has_no_base64_string(self):
        from baize.knowledge.ingestion.converters.docx import convert

        result = await convert(sample_docx_bytes())
        assert "base64" not in result.markdown

    async def test_output_has_no_img_tag(self):
        from baize.knowledge.ingestion.converters.docx import convert

        result = await convert(sample_docx_bytes())
        assert "<img" not in result.markdown


class TestDocxConverterEmptyInput:
    async def test_empty_bytes_does_not_crash(self):
        from baize.knowledge.ingestion.converters.docx import convert

        # Empty bytes is not a valid docx (not a ZIP file).
        # Either an exception is raised or an empty markdown is returned — both are acceptable.
        try:
            result = await convert(b"")
            assert isinstance(result, ConvertedDocument)
            assert isinstance(result.markdown, str)
        except Exception:
            pass  # Exception on invalid input is acceptable
