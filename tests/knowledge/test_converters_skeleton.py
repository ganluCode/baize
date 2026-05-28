"""Tests for converters package skeleton — import and class hierarchy verification."""

import pytest

from baize.knowledge.ingestion.converters import (
    ConvertedDocument,
    DocumentDecodeError,
    ScannedPdfError,
    UnsupportedFormatError,
)


def test_unsupported_format_error_is_value_error():
    assert issubclass(UnsupportedFormatError, ValueError)


def test_document_decode_error_is_value_error():
    assert issubclass(DocumentDecodeError, ValueError)


def test_scanned_pdf_error_is_exception():
    assert issubclass(ScannedPdfError, Exception)


def test_unsupported_format_error_raises():
    with pytest.raises(UnsupportedFormatError):
        raise UnsupportedFormatError("unsupported format: html")


def test_document_decode_error_raises():
    with pytest.raises(DocumentDecodeError):
        raise DocumentDecodeError("cannot decode document: invalid UTF-8")


def test_scanned_pdf_error_raises():
    with pytest.raises(ScannedPdfError):
        raise ScannedPdfError("scanned PDF: no extractable text")


def test_converted_document_fields():
    doc = ConvertedDocument(markdown="# Hello\n\nWorld", warnings=[])
    assert doc.markdown == "# Hello\n\nWorld"
    assert doc.warnings == []


def test_converted_document_with_warnings():
    doc = ConvertedDocument(markdown="# Title", warnings=["image ignored", "unsupported style"])
    assert len(doc.warnings) == 2
    assert "image ignored" in doc.warnings


def test_converted_document_default_warnings():
    doc = ConvertedDocument(markdown="text")
    assert doc.warnings == []
