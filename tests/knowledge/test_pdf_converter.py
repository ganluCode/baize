"""Unit tests for pdf.py converter (F-005 acceptance criteria)."""

from unittest.mock import MagicMock, patch

import pytest

from baize.knowledge.ingestion.converters.exceptions import (
    DocumentDecodeError,
    ScannedPdfError,
)


def _make_page(text: str) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def _make_dest(title: str) -> MagicMock:
    dest = MagicMock()
    dest.title = title
    return dest


def _make_reader(
    pages: list,
    outline: list,
    is_encrypted: bool = False,
    page_number_map: dict | None = None,
) -> MagicMock:
    reader = MagicMock()
    reader.is_encrypted = is_encrypted
    reader.pages = pages
    reader.outline = outline

    if page_number_map is not None:
        reader.get_destination_page_number.side_effect = lambda d: page_number_map[d]

    return reader


class TestPdfConverterEncrypted:
    async def test_encrypted_raises_document_decode_error(self):
        from baize.knowledge.ingestion.converters import pdf

        reader = _make_reader(pages=[], outline=[], is_encrypted=True)
        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            with pytest.raises(DocumentDecodeError) as exc_info:
                await pdf.convert(b"fake")

        assert "加密" in str(exc_info.value)


class TestPdfConverterScanned:
    async def test_blank_text_raises_scanned_pdf_error(self):
        from baize.knowledge.ingestion.converters import pdf

        reader = _make_reader(pages=[_make_page(""), _make_page("  ")], outline=[])
        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            with pytest.raises(ScannedPdfError) as exc_info:
                await pdf.convert(b"fake")

        msg = str(exc_info.value)
        assert "扫描版" in msg or "OCR" in msg

    async def test_whitespace_only_raises_scanned_pdf_error(self):
        from baize.knowledge.ingestion.converters import pdf

        reader = _make_reader(pages=[_make_page("\n\t   \n")], outline=[])
        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            with pytest.raises(ScannedPdfError):
                await pdf.convert(b"fake")

    async def test_none_extract_text_raises_scanned_pdf_error(self):
        from baize.knowledge.ingestion.converters import pdf

        page = MagicMock()
        page.extract_text.return_value = None
        reader = _make_reader(pages=[page], outline=[])
        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            with pytest.raises(ScannedPdfError):
                await pdf.convert(b"fake")


class TestPdfConverterWithOutline:
    async def test_level1_headings_use_single_hash(self):
        from baize.knowledge.ingestion.converters import pdf

        d1 = _make_dest("Chapter 1")
        d2 = _make_dest("Chapter 2")
        pages = [_make_page("content one"), _make_page("content two")]
        reader = _make_reader(pages=pages, outline=[d1, d2], page_number_map={d1: 0, d2: 1})

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        assert "# Chapter 1" in result.markdown
        assert "# Chapter 2" in result.markdown

    async def test_level2_headings_use_double_hash(self):
        from baize.knowledge.ingestion.converters import pdf

        d1 = _make_dest("Chapter 1")
        d2 = _make_dest("Section 1.1")
        pages = [_make_page("chapter content"), _make_page("section content")]
        # d2 is child of d1 → nested list
        reader = _make_reader(pages=pages, outline=[d1, [d2]], page_number_map={d1: 0, d2: 1})

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        assert "# Chapter 1" in result.markdown
        assert "## Section 1.1" in result.markdown

    async def test_level3_headings_use_triple_hash(self):
        from baize.knowledge.ingestion.converters import pdf

        d1 = _make_dest("Chapter")
        d2 = _make_dest("Section")
        d3 = _make_dest("Subsection")
        pages = [_make_page("content")]
        reader = _make_reader(
            pages=pages,
            outline=[d1, [d2, [d3]]],
            page_number_map={d1: 0, d2: 0, d3: 0},
        )

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        assert "# Chapter" in result.markdown
        assert "## Section" in result.markdown
        assert "### Subsection" in result.markdown

    async def test_heading_order_matches_outline(self):
        from baize.knowledge.ingestion.converters import pdf

        d_alpha = _make_dest("Alpha")
        d_beta = _make_dest("Beta")
        d_gamma = _make_dest("Gamma")
        pages = [_make_page("a"), _make_page("b"), _make_page("c")]
        reader = _make_reader(
            pages=pages,
            outline=[d_alpha, d_beta, d_gamma],
            page_number_map={d_alpha: 0, d_beta: 1, d_gamma: 2},
        )

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        alpha_pos = result.markdown.index("# Alpha")
        beta_pos = result.markdown.index("# Beta")
        gamma_pos = result.markdown.index("# Gamma")
        assert alpha_pos < beta_pos < gamma_pos

    async def test_deep_nesting_capped_at_h3(self):
        from baize.knowledge.ingestion.converters import pdf

        d1 = _make_dest("L1")
        d2 = _make_dest("L2")
        d3 = _make_dest("L3")
        d4 = _make_dest("L4")
        pages = [_make_page("content")]
        reader = _make_reader(
            pages=pages,
            outline=[d1, [d2, [d3, [d4]]]],
            page_number_map={d1: 0, d2: 0, d3: 0, d4: 0},
        )

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        assert "# L1" in result.markdown
        assert "## L2" in result.markdown
        assert "### L3" in result.markdown
        assert "### L4" in result.markdown  # capped at ###


class TestPdfConverterWithoutOutline:
    async def test_no_outline_returns_plain_text(self):
        from baize.knowledge.ingestion.converters import pdf

        pages = [_make_page("Plain text content here")]
        reader = _make_reader(pages=pages, outline=[])

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        assert "Plain text content here" in result.markdown
        for line in result.markdown.split("\n"):
            assert not line.startswith("#"), f"Unexpected heading: {line!r}"

    async def test_no_outline_warnings_contain_hint(self):
        from baize.knowledge.ingestion.converters import pdf

        pages = [_make_page("Some text")]
        reader = _make_reader(pages=pages, outline=[])

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        combined = " ".join(result.warnings)
        assert "无 outline" in combined or "无outline" in combined


class TestPdfConverterWarnings:
    async def test_warnings_contain_page_count(self):
        from baize.knowledge.ingestion.converters import pdf

        pages = [_make_page("text one"), _make_page("text two"), _make_page("text three")]
        reader = _make_reader(pages=pages, outline=[])

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        combined = " ".join(result.warnings)
        assert "3" in combined

    async def test_warnings_contain_char_count(self):
        from baize.knowledge.ingestion.converters import pdf

        pages = [_make_page("Hello")]
        reader = _make_reader(pages=pages, outline=[])

        with patch("baize.knowledge.ingestion.converters.pdf.PdfReader", return_value=reader):
            result = await pdf.convert(b"fake")

        combined = " ".join(result.warnings)
        assert "5" in combined  # "Hello" has 5 chars
