"""Tests for KnowledgeBase CRUD API endpoints (F-003)."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from baize.knowledge.schemas import KnowledgeBaseListResponse, KnowledgeBaseResponse
from baize.knowledge.service import KnowledgeBaseServiceError
from baize.user.deps import get_current_user
from baize.user.models import UserModel

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_USER_ID = uuid.uuid4()
_KB_ID = uuid.uuid4()

_BASE_URL = "/api/v1/knowledge-bases"
_VALID_CREATE_BODY = {
    "name": "Test KB",
    "embedding_provider": "zhipu",
    "embedding_model": "embedding-3",
    "embedding_dim": 2048,
}


def _make_user(user_id: uuid.UUID | None = None) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id or _USER_ID
    return user


def _make_kb_response(**kwargs) -> KnowledgeBaseResponse:
    defaults = dict(
        id=_KB_ID,
        user_id=_USER_ID,
        name="Test KB",
        description=None,
        embedding_provider="zhipu",
        embedding_model="embedding-3",
        embedding_dim=2048,
        settings=None,
        status="active",
        document_count=0,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return KnowledgeBaseResponse(**defaults)


@pytest.fixture
def mock_svc() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def regular_user() -> UserModel:
    return _make_user()


@pytest.fixture
async def client_auth(app, mock_svc, regular_user):
    """Client authenticated as regular user with service and provider factory mocked."""
    from baize.core.deps import get_provider_factory
    from baize.knowledge.deps import get_knowledge_base_service

    mock_pf = MagicMock()
    app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc
    app.dependency_overrides[get_current_user] = lambda: regular_user
    app.dependency_overrides[get_provider_factory] = lambda: mock_pf
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_provider_factory, None)


@pytest.fixture
async def client_no_auth(app, mock_svc):
    """Client with no authentication."""
    from baize.core.deps import get_provider_factory
    from baize.knowledge.deps import get_knowledge_base_service

    app.dependency_overrides[get_knowledge_base_service] = lambda: mock_svc
    app.dependency_overrides[get_provider_factory] = lambda: MagicMock()

    def _raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_provider_factory, None)


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge-bases
# ---------------------------------------------------------------------------


async def test_create_kb_returns_201(client_auth, mock_svc):
    mock_svc.create.return_value = _make_kb_response()

    response = await client_auth.post(_BASE_URL, json=_VALID_CREATE_BODY)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Test KB"
    assert body["id"] == str(_KB_ID)
    assert body["user_id"] == str(_USER_ID)
    assert body["document_count"] == 0


async def test_create_kb_user_id_comes_from_jwt(client_auth, mock_svc, regular_user):
    mock_svc.create.return_value = _make_kb_response()

    await client_auth.post(_BASE_URL, json=_VALID_CREATE_BODY)

    call_kwargs = mock_svc.create.call_args
    assert call_kwargs.kwargs["user_id"] == regular_user.id


async def test_create_kb_invalid_provider_returns_400(client_auth, mock_svc):
    mock_svc.create.side_effect = KnowledgeBaseServiceError(
        "LLM provider not found: 'invalid'", status_code=400
    )

    response = await client_auth.post(
        _BASE_URL,
        json={**_VALID_CREATE_BODY, "embedding_provider": "invalid"},
    )

    assert response.status_code == 400


async def test_create_kb_duplicate_name_returns_409(client_auth, mock_svc):
    mock_svc.create.side_effect = KnowledgeBaseServiceError(
        "Knowledge base with name 'Test KB' already exists.", status_code=409
    )

    response = await client_auth.post(_BASE_URL, json=_VALID_CREATE_BODY)

    assert response.status_code == 409


async def test_create_kb_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.post(_BASE_URL, json=_VALID_CREATE_BODY)
    assert response.status_code == 401


async def test_create_kb_missing_required_field_returns_422(client_auth, mock_svc):
    response = await client_auth.post(
        _BASE_URL,
        json={"name": "Test KB"},  # missing embedding_provider, embedding_model, embedding_dim
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/knowledge-bases
# ---------------------------------------------------------------------------


async def test_list_kbs_returns_200_with_items(client_auth, mock_svc):
    mock_svc.list.return_value = KnowledgeBaseListResponse(
        items=[_make_kb_response()], total=1
    )

    response = await client_auth.get(_BASE_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == str(_KB_ID)


async def test_list_kbs_default_status_is_active(client_auth, mock_svc, regular_user):
    mock_svc.list.return_value = KnowledgeBaseListResponse(items=[], total=0)

    await client_auth.get(_BASE_URL)

    call_kwargs = mock_svc.list.call_args
    assert call_kwargs.kwargs["status"] == "active"
    assert call_kwargs.kwargs["user_id"] == regular_user.id


async def test_list_kbs_passes_limit_offset(client_auth, mock_svc):
    mock_svc.list.return_value = KnowledgeBaseListResponse(items=[], total=0)

    await client_auth.get(f"{_BASE_URL}?limit=5&offset=10")

    call_kwargs = mock_svc.list.call_args
    assert call_kwargs.kwargs["limit"] == 5
    assert call_kwargs.kwargs["offset"] == 10


async def test_list_kbs_passes_custom_status(client_auth, mock_svc):
    mock_svc.list.return_value = KnowledgeBaseListResponse(items=[], total=0)

    await client_auth.get(f"{_BASE_URL}?status=deleted")

    call_kwargs = mock_svc.list.call_args
    assert call_kwargs.kwargs["status"] == "deleted"


async def test_list_kbs_empty_returns_empty_list(client_auth, mock_svc):
    mock_svc.list.return_value = KnowledgeBaseListResponse(items=[], total=0)

    response = await client_auth.get(_BASE_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []


async def test_list_kbs_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.get(_BASE_URL)
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/knowledge-bases/{kb_id}
# ---------------------------------------------------------------------------


async def test_get_kb_returns_200(client_auth, mock_svc):
    mock_svc.get.return_value = _make_kb_response()

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}")

    assert response.status_code == 200
    assert response.json()["id"] == str(_KB_ID)


async def test_get_kb_calls_service_with_correct_args(client_auth, mock_svc, regular_user):
    mock_svc.get.return_value = _make_kb_response()

    await client_auth.get(f"{_BASE_URL}/{_KB_ID}")

    mock_svc.get.assert_awaited_once_with(kb_id=_KB_ID, user_id=regular_user.id)


async def test_get_kb_wrong_user_returns_403(client_auth, mock_svc):
    mock_svc.get.side_effect = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}")

    assert response.status_code == 403


async def test_get_kb_not_found_returns_404(client_auth, mock_svc):
    mock_svc.get.side_effect = KnowledgeBaseServiceError(
        "Knowledge base not found.", status_code=404
    )

    response = await client_auth.get(f"{_BASE_URL}/{_KB_ID}")

    assert response.status_code == 404


async def test_get_kb_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.get(f"{_BASE_URL}/{_KB_ID}")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /api/v1/knowledge-bases/{kb_id}
# ---------------------------------------------------------------------------


async def test_delete_kb_returns_204(client_auth, mock_svc):
    mock_svc.soft_delete.return_value = None

    response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}")

    assert response.status_code == 204


async def test_delete_kb_calls_service_with_correct_args(client_auth, mock_svc, regular_user):
    mock_svc.soft_delete.return_value = None

    await client_auth.delete(f"{_BASE_URL}/{_KB_ID}")

    mock_svc.soft_delete.assert_awaited_once_with(kb_id=_KB_ID, user_id=regular_user.id)


async def test_delete_kb_wrong_user_returns_403(client_auth, mock_svc):
    mock_svc.soft_delete.side_effect = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )

    response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}")

    assert response.status_code == 403


async def test_delete_kb_not_found_returns_404(client_auth, mock_svc):
    mock_svc.soft_delete.side_effect = KnowledgeBaseServiceError(
        "Knowledge base not found.", status_code=404
    )

    response = await client_auth.delete(f"{_BASE_URL}/{_KB_ID}")

    assert response.status_code == 404


async def test_delete_kb_without_auth_returns_401(client_no_auth):
    response = await client_no_auth.delete(f"{_BASE_URL}/{_KB_ID}")
    assert response.status_code == 401
