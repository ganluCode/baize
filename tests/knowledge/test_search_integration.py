"""Search API end-to-end integration tests (F-014).

Tests:
- Full KB lifecycle: Create KB → JSON doc → wait ingested → search → delete doc → delete KB
- OpenAPI schema: 10 /api/v1/knowledge-bases operations registered
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.knowledge.conftest import wait_for_document_status

_BASE = "/api/v1/knowledge-bases"
_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def _make_ingested_doc(doc_id: uuid.UUID, kb_id: uuid.UUID, title: str):
    """Create a KnowledgeDocumentModel instance with status='ingested' for mock DB responses."""
    from baize.knowledge.models import KnowledgeDocumentModel

    now = datetime.now(UTC)
    return KnowledgeDocumentModel(
        id=doc_id,
        kb_id=kb_id,
        title=title,
        source_type="markdown",
        source_uri=None,
        content_hash="sha256-integration-test-hash",
        doc_metadata={},
        chunk_count=2,
        status="ingested",
        error_message=None,
        created_at=now,
        updated_at=now,
    )


async def test_full_knowledge_base_lifecycle(
    api_client,
    knowledge_app,
    mock_db_session,
    create_test_user_and_token,
):
    """Integration: Create KB → JSON doc → wait ingested → search → delete doc → delete KB."""
    from baize.knowledge.deps import get_knowledge_base_service, get_retrieval_service
    from baize.knowledge.retrieval.types import ScoredChunk
    from baize.knowledge.schemas import KnowledgeBaseResponse

    user_id, token = create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    kb_id = uuid.uuid4()
    doc_title = "Integration Test Document"
    now = datetime.now(UTC)

    # ── Mock KB service ────────────────────────────────────────────────────────
    kb_response = KnowledgeBaseResponse(
        id=kb_id,
        user_id=user_id,
        name="E2E Test KB",
        description=None,
        embedding_provider="test-provider",
        embedding_model="test-embed-model",
        embedding_dim=512,
        status="active",
        settings=None,
        document_count=0,
        created_at=now,
        updated_at=now,
    )
    mock_kb_svc = AsyncMock()
    mock_kb_svc.create.return_value = kb_response
    mock_kb_svc.get.return_value = MagicMock(id=kb_id, user_id=user_id)
    mock_kb_svc.soft_delete.return_value = None

    # ── Mock retrieval service (configured with doc_id after doc creation) ─────
    mock_retrieval_svc = AsyncMock()

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_kb_svc
    knowledge_app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_svc

    try:
        # ── Step 1: Create KB ──────────────────────────────────────────────────
        r = await api_client.post(
            _BASE,
            json={
                "name": "E2E Test KB",
                "embedding_provider": "test-provider",
                "embedding_model": "test-embed-model",
                "embedding_dim": 512,
            },
            headers=headers,
        )
        assert r.status_code == 201
        assert r.json()["id"] == str(kb_id)

        # ── Step 2: Create document via JSON (background task patched) ─────────
        async def _fake_ingest(*, doc_id, kb_id, content):
            pass  # Simulate background ingestion completing without errors

        with patch("baize.knowledge.api._bg_ingest_document", new=_fake_ingest):
            r = await api_client.post(
                f"{_BASE}/{kb_id}/documents",
                json={
                    "title": doc_title,
                    "content": "# Integration Test\n\n## Section\n\nContent for E2E testing.",
                    "source_type": "markdown",
                },
                headers=headers,
            )

        assert r.status_code == 202
        doc_body = r.json()
        assert doc_body["status"] == "pending"
        actual_doc_id = uuid.UUID(doc_body["id"])
        assert doc_body["title"] == doc_title

        # ── Step 3: Wait for document status 'ingested' ────────────────────────
        # Configure mock DB to return an ingested document for GET /documents/{doc_id}
        ingested_doc = _make_ingested_doc(actual_doc_id, kb_id, doc_title)

        get_doc_result = MagicMock()
        get_doc_result.scalar_one_or_none.return_value = ingested_doc
        mock_db_session.execute = AsyncMock(return_value=get_doc_result)

        doc_data = await wait_for_document_status(
            api_client, kb_id, actual_doc_id, "ingested", token, timeout=5
        )
        assert doc_data["status"] == "ingested"
        assert doc_data["title"] == doc_title

        # ── Step 4: Search ──────────────────────────────────────────────────────
        chunk = ScoredChunk(
            chunk_id=uuid.uuid4(),
            doc_id=actual_doc_id,
            kb_id=kb_id,
            content="Integration test chunk content",
            section_path=["Integration Test", "Section"],
            level=2,
            parent_chunk_id=None,
            score=0.92,
            rank=1,
            sources=["bm25", "vector"],
            bm25_rank=1,
            vector_rank=1,
            bm25_score=0.8,
            vector_score=0.95,
        )
        mock_retrieval_svc.search.return_value = [chunk]

        # Configure mock DB to return doc title for search result enrichment
        title_result = MagicMock()
        title_result.all.return_value = [(actual_doc_id, doc_title)]
        mock_db_session.execute = AsyncMock(return_value=title_result)

        r = await api_client.post(
            f"{_BASE}/{kb_id}/search",
            json={"query": "integration test section", "top_k": 5},
            headers=headers,
        )
        assert r.status_code == 200
        search_body = r.json()
        assert search_body["total"] == 1
        assert isinstance(search_body["latency_ms"], int)
        assert search_body["latency_ms"] >= 0
        assert len(search_body["results"]) == 1
        result = search_body["results"][0]
        assert result["doc_title"] == doc_title
        assert result["doc_id"] == str(actual_doc_id)

        # ── Step 5: Delete document → 204 (verifies DB cleanup) ────────────────
        # delete_document endpoint executes 4 DB statements:
        #   1. SELECT document (verify exists)
        #   2. SELECT chunks (collect qdrant_point_ids)
        #   3. DELETE FROM chunks WHERE doc_id = ...
        #   4. DELETE FROM documents WHERE id = ...
        delete_call_num = 0

        def _del_execute(stmt, *args, **kwargs):
            nonlocal delete_call_num
            delete_call_num += 1
            result = MagicMock()
            if delete_call_num == 1:
                # SELECT document — must exist for 204 (not 404)
                result.scalar_one_or_none.return_value = ingested_doc
            elif delete_call_num == 2:
                # SELECT chunks — no chunks in mock (no Qdrant delete needed)
                result.scalars.return_value.all.return_value = []
            # Calls 3 & 4 (DELETE statements): return value ignored by endpoint
            return result

        mock_db_session.execute = AsyncMock(side_effect=_del_execute)
        mock_db_session.commit.reset_mock()

        r = await api_client.delete(
            f"{_BASE}/{kb_id}/documents/{actual_doc_id}",
            headers=headers,
        )
        assert r.status_code == 204

        # Both DELETE statements and commit must have been issued
        assert delete_call_num == 4, (
            f"Expected 4 DB execute calls (select doc, select chunks, "
            f"delete chunks, delete doc), got {delete_call_num}"
        )
        assert mock_db_session.commit.called, "DB commit must be called after document deletion"

        # ── Step 6: Delete KB → 204 ────────────────────────────────────────────
        r = await api_client.delete(
            f"{_BASE}/{kb_id}",
            headers=headers,
        )
        assert r.status_code == 204
        mock_kb_svc.soft_delete.assert_called_once()

    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)
        knowledge_app.dependency_overrides.pop(get_retrieval_service, None)


async def test_search_doc_title_matches_database_title(
    api_client,
    knowledge_app,
    mock_db_session,
    create_test_user_and_token,
):
    """CitationResponse.doc_title matches the document title stored in the database (non-empty)."""
    from baize.knowledge.deps import get_knowledge_base_service, get_retrieval_service
    from baize.knowledge.retrieval.types import ScoredChunk

    user_id, token = create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    expected_title = "My Knowledge Document"

    mock_kb_svc = AsyncMock()
    mock_kb_svc.get.return_value = MagicMock(id=kb_id, user_id=user_id)

    chunk = ScoredChunk(
        chunk_id=uuid.uuid4(),
        doc_id=doc_id,
        kb_id=kb_id,
        content="Sample chunk",
        section_path=["Intro"],
        level=1,
        parent_chunk_id=None,
        score=0.88,
        rank=1,
        sources=["vector"],
        bm25_rank=None,
        vector_rank=1,
        bm25_score=None,
        vector_score=0.88,
    )
    mock_retrieval_svc = AsyncMock()
    mock_retrieval_svc.search.return_value = [chunk]

    # DB returns the document title matching the stored record
    title_result = MagicMock()
    title_result.all.return_value = [(doc_id, expected_title)]
    mock_db_session.execute = AsyncMock(return_value=title_result)

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_kb_svc
    knowledge_app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_svc

    try:
        r = await api_client.post(
            f"{_BASE}/{kb_id}/search",
            json={"query": "knowledge query", "top_k": 3},
            headers=headers,
        )
        assert r.status_code == 200
        body = r.json()

        assert len(body["results"]) == 1
        citation = body["results"][0]
        assert citation["doc_title"] == expected_title
        assert citation["doc_title"] != ""  # non-empty string

    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)
        knowledge_app.dependency_overrides.pop(get_retrieval_service, None)


async def test_search_latency_ms_is_non_negative_integer(
    api_client,
    knowledge_app,
    mock_db_session,
    create_test_user_and_token,
):
    """SearchResponse.latency_ms is a non-negative integer measured in milliseconds."""
    from baize.knowledge.deps import get_knowledge_base_service, get_retrieval_service

    user_id, token = create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    kb_id = uuid.uuid4()

    mock_kb_svc = AsyncMock()
    mock_kb_svc.get.return_value = MagicMock(id=kb_id, user_id=user_id)

    mock_retrieval_svc = AsyncMock()
    mock_retrieval_svc.search.return_value = []

    empty_result = MagicMock()
    empty_result.all.return_value = []
    mock_db_session.execute = AsyncMock(return_value=empty_result)

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_kb_svc
    knowledge_app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_svc

    try:
        r = await api_client.post(
            f"{_BASE}/{kb_id}/search",
            json={"query": "latency test"},
            headers=headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["latency_ms"], int)
        assert body["latency_ms"] >= 0

    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)
        knowledge_app.dependency_overrides.pop(get_retrieval_service, None)


async def test_openapi_paths_include_10_knowledge_base_operations(api_client):
    """GET /openapi.json registers exactly 10 /api/v1/knowledge-bases operations.

    Operations:
      KB (4):   POST create, GET list, GET single, DELETE soft-delete
      Doc (5):  POST json, POST upload, GET list, GET single, DELETE hard-delete
      Search (1): POST search
    """
    r = await api_client.get("/openapi.json")
    assert r.status_code == 200

    paths = r.json().get("paths", {})
    kb_operations = sum(
        sum(1 for method in path_item if method in _HTTP_METHODS)
        for path, path_item in paths.items()
        if "/api/v1/knowledge-bases" in path
    )

    assert kb_operations == 10, (
        f"Expected 10 knowledge-bases operations in OpenAPI schema, "
        f"got {kb_operations}. "
        f"KB paths found: {[p for p in paths if '/api/v1/knowledge-bases' in p]}"
    )
