"""Verification tests for knowledge API test infrastructure (F-010).

These tests confirm that the fixtures provided by tests/knowledge/conftest.py
are correctly set up and usable by downstream test suites (F-011 through F-014).
"""

import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from tests.knowledge.conftest import wait_for_document_status


# ─── create_test_user_and_token ──────────────────────────────────────────────


async def test_create_test_user_and_token_returns_uuid_and_str(
    create_test_user_and_token,
):
    """Factory returns (UUID, non-empty str) on each call."""
    user_id, token = create_test_user_and_token()
    assert isinstance(user_id, uuid.UUID)
    assert isinstance(token, str)
    assert len(token) > 0


async def test_two_users_have_distinct_ids_and_tokens(create_test_user_and_token):
    """Two calls produce different user IDs and tokens."""
    uid_a, tok_a = create_test_user_and_token("alice")
    uid_b, tok_b = create_test_user_and_token("bob")
    assert uid_a != uid_b
    assert tok_a != tok_b


async def test_user_role_is_preserved(create_test_user_and_token, user_registry):
    """User role passed to the factory is stored on the UserModel mock."""
    user_id, token = create_test_user_and_token(role="admin")
    user = user_registry[token]
    assert user.role == "admin"
    assert user.id == user_id


# ─── Authentication ───────────────────────────────────────────────────────────


async def test_unauthenticated_request_returns_401(api_client: AsyncClient):
    """GET /knowledge-bases without a token returns 401 Not authenticated."""
    response = await api_client.get("/api/v1/knowledge-bases")
    assert response.status_code == 401


async def test_bearer_token_bypasses_401(
    api_client: AsyncClient,
    create_test_user_and_token,
    knowledge_app,
):
    """A registered Bearer token results in a non-401 response."""
    from baize.knowledge.deps import get_knowledge_base_service
    from baize.knowledge.schemas import KnowledgeBaseListResponse

    user_id, token = create_test_user_and_token()

    mock_svc = AsyncMock()
    mock_svc.list.return_value = KnowledgeBaseListResponse(items=[], total=0)
    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc

    try:
        response = await api_client.get(
            "/api/v1/knowledge-bases",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code != 401
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)


async def test_unknown_token_returns_401(api_client: AsyncClient):
    """An unregistered Bearer token returns 401."""
    response = await api_client.get(
        "/api/v1/knowledge-bases",
        headers={"Authorization": "Bearer this-token-is-not-registered"},
    )
    assert response.status_code == 401


# ─── mock_db_session isolation ───────────────────────────────────────────────


async def test_mock_db_session_starts_clean(mock_db_session):
    """Session mock has no prior call history at test start."""
    assert mock_db_session.add.call_count == 0
    assert mock_db_session.commit.call_count == 0


async def test_mock_db_session_execute_returns_empty_by_default(mock_db_session):
    """Default execute() returns a result with scalar_one_or_none() == None."""
    from unittest.mock import MagicMock

    result = await mock_db_session.execute(MagicMock())
    assert result.scalar_one_or_none() is None
    assert result.one_or_none() is None
    assert result.scalar_one() == 0


async def test_mock_db_session_is_isolated_between_tests_a(mock_db_session):
    """Add a call in test A — will not appear in test B."""
    mock_db_session.add("marker_a")
    assert mock_db_session.add.call_count == 1


async def test_mock_db_session_is_isolated_between_tests_b(mock_db_session):
    """Previous test's calls do not leak into this test (fresh mock per test)."""
    assert mock_db_session.add.call_count == 0


# ─── mock_qdrant ─────────────────────────────────────────────────────────────


async def test_mock_qdrant_supports_expected_operations(mock_qdrant):
    """Mock Qdrant client has all async methods expected by the knowledge module."""
    await mock_qdrant.upsert()
    await mock_qdrant.delete()
    await mock_qdrant.query_points()
    await mock_qdrant.search()
    await mock_qdrant.create_collection()
    await mock_qdrant.collection_exists()
    # All calls recorded
    assert mock_qdrant.upsert.called
    assert mock_qdrant.delete.called


# ─── wait_for_document_status ────────────────────────────────────────────────


async def test_wait_for_document_status_raises_assertion_error_on_timeout(
    api_client: AsyncClient,
    create_test_user_and_token,
):
    """wait_for_document_status raises AssertionError when status never matches."""
    _, token = create_test_user_and_token()
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    with pytest.raises(AssertionError, match="did not reach status"):
        await wait_for_document_status(
            client=api_client,
            kb_id=kb_id,
            doc_id=doc_id,
            expected_status="ingested",
            token=token,
            timeout=1,
        )


async def test_wait_for_document_status_returns_data_when_matched(
    api_client: AsyncClient,
    create_test_user_and_token,
    knowledge_app,
    mock_db_session,
):
    """wait_for_document_status returns document dict when status matches."""
    from unittest.mock import MagicMock
    from baize.knowledge.deps import get_knowledge_base_service
    from baize.knowledge.models import KnowledgeDocumentModel

    user_id, token = create_test_user_and_token()
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    # Mock: KB exists (for svc.get), and doc exists with status=ingested
    mock_svc = AsyncMock()
    mock_svc.get.return_value = MagicMock(id=kb_id, user_id=user_id)

    doc = MagicMock(spec=KnowledgeDocumentModel)
    doc.id = doc_id
    doc.kb_id = kb_id
    doc.title = "Test Doc"
    doc.status = "ingested"
    doc.source_type = "markdown"
    doc.source_uri = None
    doc.content_hash = "abc"
    doc.doc_metadata = {}
    doc.chunk_count = 2
    doc.error_message = None

    from datetime import UTC, datetime
    _now = datetime(2026, 1, 1, tzinfo=UTC)
    doc.created_at = _now
    doc.updated_at = _now

    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = doc
    mock_db_session.execute = AsyncMock(return_value=doc_result)

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc

    try:
        data = await wait_for_document_status(
            client=api_client,
            kb_id=kb_id,
            doc_id=doc_id,
            expected_status="ingested",
            token=token,
            timeout=5,
        )
        assert data["status"] == "ingested"
        assert data["id"] == str(doc_id)
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)
