"""Authentication and authorization tests for Knowledge Base API (F-011).

Tests:
- 401 for every endpoint when no Authorization header is provided
- 403 when User B's valid JWT is used to access a KB owned by User A
"""

import uuid
from unittest.mock import AsyncMock

import pytest

from baize.knowledge.service import KnowledgeBaseServiceError

_BASE_URL = "/api/v1/knowledge-bases"
_VALID_KB_BODY = {
    "name": "Test KB",
    "embedding_provider": "zhipu",
    "embedding_model": "embedding-3",
    "embedding_dim": 2048,
}
_VALID_DOC_BODY = {
    "title": "Test Doc",
    "content": "# Hello\n\nWorld",
    "source_type": "markdown",
}
_VALID_SEARCH_BODY = {"query": "hello world"}


# ─── 401 — unauthenticated requests ─────────────────────────────────────────


async def test_create_kb_without_auth_returns_401(api_client):
    """POST /knowledge-bases without Authorization header returns 401."""
    response = await api_client.post(_BASE_URL, json=_VALID_KB_BODY)
    assert response.status_code == 401


async def test_get_kb_without_auth_returns_401(api_client):
    """GET /knowledge-bases/{kb_id} without Authorization header returns 401."""
    response = await api_client.get(f"{_BASE_URL}/{uuid.uuid4()}")
    assert response.status_code == 401


async def test_create_document_without_auth_returns_401(api_client):
    """POST /knowledge-bases/{kb_id}/documents without Authorization header returns 401."""
    response = await api_client.post(
        f"{_BASE_URL}/{uuid.uuid4()}/documents",
        json=_VALID_DOC_BODY,
    )
    assert response.status_code == 401


async def test_search_without_auth_returns_401(api_client):
    """POST /knowledge-bases/{kb_id}/search without Authorization header returns 401."""
    response = await api_client.post(
        f"{_BASE_URL}/{uuid.uuid4()}/search",
        json=_VALID_SEARCH_BODY,
    )
    assert response.status_code == 401


# ─── 403 — cross-user access ─────────────────────────────────────────────────


@pytest.fixture
def two_users(create_test_user_and_token):
    """Create two distinct users; return (user_a_id, token_a, user_b_id, token_b)."""
    user_a_id, token_a = create_test_user_and_token("alice")
    user_b_id, token_b = create_test_user_and_token("bob")
    return user_a_id, token_a, user_b_id, token_b


@pytest.fixture
def mock_svc_403():
    """KnowledgeBaseService mock that raises 403 on any KB ownership check."""
    svc = AsyncMock()
    _forbidden = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )
    svc.get.side_effect = _forbidden
    svc.soft_delete.side_effect = _forbidden
    return svc


async def test_get_kb_cross_user_returns_403(
    api_client, knowledge_app, two_users, mock_svc_403
):
    """User B's valid JWT attempting GET on User A's KB returns 403."""
    from baize.knowledge.deps import get_knowledge_base_service

    _, _, _, token_b = two_users
    kb_id = uuid.uuid4()

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc_403
    try:
        response = await api_client.get(
            f"{_BASE_URL}/{kb_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response.status_code == 403
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)


async def test_delete_kb_cross_user_returns_403(
    api_client, knowledge_app, two_users, mock_svc_403
):
    """User B's valid JWT attempting DELETE on User A's KB returns 403."""
    from baize.knowledge.deps import get_knowledge_base_service

    _, _, _, token_b = two_users
    kb_id = uuid.uuid4()

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc_403
    try:
        response = await api_client.delete(
            f"{_BASE_URL}/{kb_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response.status_code == 403
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)


async def test_search_cross_user_returns_403(
    api_client, knowledge_app, two_users, mock_svc_403
):
    """User B's valid JWT attempting search on User A's KB returns 403."""
    from baize.knowledge.deps import get_knowledge_base_service

    _, _, _, token_b = two_users
    kb_id = uuid.uuid4()

    knowledge_app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc_403
    try:
        response = await api_client.post(
            f"{_BASE_URL}/{kb_id}/search",
            json=_VALID_SEARCH_BODY,
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response.status_code == 403
    finally:
        knowledge_app.dependency_overrides.pop(get_knowledge_base_service, None)
