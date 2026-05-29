"""Synchronous phase unit tests for document endpoints (F-012).

Tests:
- POST /{kb_id}/documents (JSON): 202 with status='pending'
- POST /{kb_id}/documents (JSON): 409 on duplicate content (existing_doc_id in detail)
- POST /{kb_id}/documents/upload: 415 on unsupported format (.doc), detail lists 4 formats
- POST /{kb_id}/documents/upload: 413 on file > 10MB
- POST /{kb_id}/documents/upload: 400 on GBK-encoded .txt (detail mentions utf-8)
- POST /{kb_id}/documents/upload: 400 on encrypted PDF
- POST /{kb_id}/documents/upload: title auto-generated from filename when not provided
- POST /{kb_id}/documents/upload: 202 with status='pending' for valid .md file
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_BASE = "/api/v1/knowledge-bases"

_VALID_JSON_BODY = {
    "title": "Test Document",
    "content": "# Hello\n\nThis is test content.",
    "source_type": "markdown",
}


@pytest.fixture
def kb_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def auth_token(create_test_user_and_token) -> str:
    _, token = create_test_user_and_token()
    return token


@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def mock_svc_ok() -> AsyncMock:
    """KnowledgeBaseService mock where get() returns successfully (does not raise)."""
    svc = AsyncMock()
    svc.get.return_value = MagicMock()
    return svc


# ─── JSON document endpoint ──────────────────────────────────────────────────


async def test_create_document_json_returns_202_pending(
    api_client, knowledge_app, kb_id, auth_headers, mock_svc_ok
):
    """Valid JSON body immediately returns 202 with status='pending'."""
    from baize.knowledge.deps import get_knowledge_base_service

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc_ok
    try:
        response = await api_client.post(
            f"{_BASE}/{kb_id}/documents",
            json=_VALID_JSON_BODY,
            headers=auth_headers,
        )
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)

    assert response.status_code == 202
    assert response.json()["status"] == "pending"


async def test_create_document_json_duplicate_returns_409_with_existing_doc_id(
    api_client, knowledge_app, mock_db_session, kb_id, auth_headers, mock_svc_ok
):
    """Uploading content already ingested in the same KB returns 409 with existing_doc_id."""
    from baize.knowledge.deps import get_knowledge_base_service

    existing_doc_id = uuid.uuid4()
    existing_doc = MagicMock()
    existing_doc.id = existing_doc_id

    dup_result = MagicMock()
    dup_result.scalar_one_or_none.return_value = existing_doc
    mock_db_session.execute.return_value = dup_result

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc_ok
    try:
        response = await api_client.post(
            f"{_BASE}/{kb_id}/documents",
            json=_VALID_JSON_BODY,
            headers=auth_headers,
        )
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)

    assert response.status_code == 409
    message = response.json()["message"]
    assert str(existing_doc_id) in message


# ─── Multipart upload — pre-check failures ────────────────────────────────────


async def test_upload_doc_format_returns_415_with_supported_formats(
    api_client, kb_id, auth_headers
):
    """.doc file extension returns 415; detail lists all four supported formats."""
    response = await api_client.post(
        f"{_BASE}/{kb_id}/documents/upload",
        files={"file": ("report.doc", b"fake content", "application/msword")},
        headers=auth_headers,
    )

    assert response.status_code == 415
    message = response.json()["message"]
    for fmt in (".md", ".txt", ".docx", ".pdf"):
        assert fmt in message


async def test_upload_oversized_file_returns_413(
    api_client, kb_id, auth_headers
):
    """File exceeding 10MB limit returns 413."""
    large_content = b"a" * (10 * 1024 * 1024 + 1)
    response = await api_client.post(
        f"{_BASE}/{kb_id}/documents/upload",
        files={"file": ("large.md", large_content, "text/markdown")},
        headers=auth_headers,
    )

    assert response.status_code == 413


async def test_upload_gbk_txt_returns_400_mentioning_utf8(
    api_client, kb_id, auth_headers
):
    """GBK-encoded .txt file returns 400; detail mentions utf-8."""
    gbk_bytes = "你好世界".encode("gbk")
    response = await api_client.post(
        f"{_BASE}/{kb_id}/documents/upload",
        files={"file": ("notes.txt", gbk_bytes, "text/plain")},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "utf-8" in response.json()["message"].lower()


async def test_upload_encrypted_pdf_returns_400(
    api_client, kb_id, auth_headers
):
    """Encrypted PDF (mocked PdfReader.is_encrypted=True) returns 400."""
    with patch("baize.knowledge.api.PdfReader") as mock_pdf_cls:
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        mock_pdf_cls.return_value = mock_reader

        response = await api_client.post(
            f"{_BASE}/{kb_id}/documents/upload",
            files={"file": ("secret.pdf", b"%PDF-1.4 fake", "application/pdf")},
            headers=auth_headers,
        )

    assert response.status_code == 400


# ─── Multipart upload — success path ─────────────────────────────────────────


async def test_upload_title_derived_from_filename_when_not_provided(
    api_client, knowledge_app, kb_id, auth_headers, mock_svc_ok
):
    """When title is omitted, response title equals filename without extension."""
    from baize.knowledge.deps import get_knowledge_base_service

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc_ok
    try:
        response = await api_client.post(
            f"{_BASE}/{kb_id}/documents/upload",
            files={"file": ("my_document.md", b"# Hello World", "text/markdown")},
            headers=auth_headers,
        )
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)

    assert response.status_code == 202
    assert response.json()["title"] == "my_document"


async def test_upload_valid_md_returns_202_pending(
    api_client, knowledge_app, kb_id, auth_headers, mock_svc_ok
):
    """Valid .md file upload returns 202 with status='pending'."""
    from baize.knowledge.deps import get_knowledge_base_service

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc_ok
    try:
        response = await api_client.post(
            f"{_BASE}/{kb_id}/documents/upload",
            files={"file": ("guide.md", b"# Guide\n\nContent here.", "text/markdown")},
            headers=auth_headers,
        )
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
