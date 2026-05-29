"""Helper functions for reading test fixture binary files."""
from pathlib import Path

_FIXTURES_DIR = Path(__file__).parent


def fixture_bytes(filename: str) -> bytes:
    """Return raw bytes of a fixture file located in this directory."""
    return (_FIXTURES_DIR / filename).read_bytes()


def sample_docx_bytes() -> bytes:
    """Return bytes of the minimal valid sample.docx fixture."""
    return fixture_bytes("sample.docx")


def sample_pdf_bytes() -> bytes:
    """Return bytes of the minimal valid sample.pdf fixture."""
    return fixture_bytes("sample.pdf")
