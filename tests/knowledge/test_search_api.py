"""Tests for Knowledge Base Search API endpoint (F-007).

Covers:
  POST /{kb_id}/search — hybrid search returning CitationResponse list
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from baize.knowledge.retrieval.types import ScoredChunk
from baize.knowledge.service import KnowledgeBaseServiceError
from baize.user.models import UserModel

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_USER_ID = uuid.uuid4()
_KB_ID = uuid.uuid4()
_DOC_ID = uuid.uuid4()
_CHUNK_ID = uuid.uuid4()
_BASE_URL = "/api/v1/knowledge-bases"


def _make_user(user_id: uuid.UUID | None = None) -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = user_id or _USER_ID
    return user


def _make_scored_chunk(
    chunk_id: uuid.UUID | None = None,
    doc_id: uuid.UUID | None = None,
    score: float = 0.85,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id or _CHUNK_ID,
        doc_id=doc_id or _DOC_ID,
        kb_id=_KB_ID,
        content="Some chunk content",
        section_path=["Introduction", "Overview"],
        level=2,
        parent_chunk_id=None,
        score=score,
        rank=1,
        sources=["source1"],
        bm25_rank=1,
        vector_rank=1,
        bm25_score=0.7,
        vector_score=0.9,
    )


# ────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────


@pytest.fixture
def mock_kb_svc() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_retrieval_svc() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def regular_user() -> UserModel:
    return _make_user()


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
async def client_auth(app, mock_kb_svc, mock_retrieval_svc, mock_db_session, regular_user):
    from baize.core.database import get_db
    from baize.knowledge.deps import get_knowledge_base_service, get_retrieval_service
    from baize.user.deps import get_current_user

    app.dependency_overrides[get_knowledge_base_service] = lambda: mock_kb_svc
    app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_svc
    app.dependency_overrides[get_current_user] = lambda: regular_user
    app.dependency_overrides[get_db] = lambda: mock_db_session
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_retrieval_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def client_no_auth(app, mock_kb_svc, mock_retrieval_svc, mock_db_session):
    from baize.core.database import get_db
    from baize.knowledge.deps import get_knowledge_base_service, get_retrieval_service
    from baize.user.deps import get_current_user

    app.dependency_overrides[get_knowledge_base_service] = lambda: mock_kb_svc
    app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_svc
    app.dependency_overrides[get_db] = lambda: mock_db_session

    def _raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = _raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_retrieval_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)


# ────────────────────────────────────────
# POST /{kb_id}/search — success paths
# ────────────────────────────────────────


async def test_search_returns_200_with_citation_responses(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """Search returns 200 with correct CitationResponse fields including doc_title."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    chunk = _make_scored_chunk()
    mock_retrieval_svc.search = AsyncMock(return_value=[chunk])

    # Mock DB execute to return doc title
    title_row = MagicMock()
    title_row.all.return_value = [(_DOC_ID, "My Document Title")]
    db_result = MagicMock()
    db_result.all = title_row.all
    mock_db_session.execute = AsyncMock(return_value=db_result)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "test query", "top_k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "test query"
    assert body["total"] == 1
    assert isinstance(body["latency_ms"], int)
    assert body["latency_ms"] >= 0
    assert len(body["results"]) == 1

    result = body["results"][0]
    assert result["doc_title"] == "My Document Title"
    assert result["chunk_id"] == str(_CHUNK_ID)
    assert result["doc_id"] == str(_DOC_ID)
    assert result["section_path"] == ["Introduction", "Overview"]
    assert result["score"] == pytest.approx(0.85, abs=0.001)
    assert result["sources"] == ["source1"]


async def test_search_latency_ms_is_positive_integer(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """latency_ms field in SearchResponse must be a non-negative integer."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    mock_retrieval_svc.search = AsyncMock(return_value=[])
    mock_db_session.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[])))

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "some query"},
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["latency_ms"], int)
    assert body["latency_ms"] >= 0


async def test_search_empty_results(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """Search with no matching chunks returns empty results list with total=0."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    mock_retrieval_svc.search = AsyncMock(return_value=[])
    mock_db_session.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[])))

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "no results query", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "no results query"
    assert body["total"] == 0
    assert body["results"] == []


async def test_search_passes_top_k_and_include_parents_to_service(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """SearchRequest fields are forwarded correctly to RetrievalService.search()."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    mock_retrieval_svc.search = AsyncMock(return_value=[])
    mock_db_session.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[])))

    await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "parent search", "top_k": 10, "include_parents": True},
    )

    call_kwargs = mock_retrieval_svc.search.call_args.kwargs
    assert call_kwargs["kb_id"] == _KB_ID
    assert call_kwargs["query"] == "parent search"
    assert call_kwargs["top_k"] == 10
    assert call_kwargs["include_parents"] is True


async def test_search_doc_title_missing_falls_back_to_empty_string(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """doc_title defaults to empty string when doc is not found in DB (race condition)."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    unknown_doc_id = uuid.uuid4()
    chunk = _make_scored_chunk(doc_id=unknown_doc_id)
    mock_retrieval_svc.search = AsyncMock(return_value=[chunk])

    # DB returns no rows for doc_id lookup
    db_result = MagicMock()
    db_result.all = MagicMock(return_value=[])
    mock_db_session.execute = AsyncMock(return_value=db_result)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "query"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["doc_title"] == ""


async def test_search_multiple_chunks_same_doc_fetches_title_once(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """Multiple chunks from the same document share the same doc_title lookup."""
    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    chunk1 = _make_scored_chunk(chunk_id=uuid.uuid4(), score=0.9)
    chunk2 = _make_scored_chunk(chunk_id=uuid.uuid4(), score=0.8)
    mock_retrieval_svc.search = AsyncMock(return_value=[chunk1, chunk2])

    db_result = MagicMock()
    db_result.all = MagicMock(return_value=[(_DOC_ID, "Shared Doc Title")])
    mock_db_session.execute = AsyncMock(return_value=db_result)

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "multi-chunk"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    for result in body["results"]:
        assert result["doc_title"] == "Shared Doc Title"


# ────────────────────────────────────────
# POST /{kb_id}/search — error paths
# ────────────────────────────────────────


async def test_search_returns_404_when_kb_not_found(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """Returns 404 when KB does not exist."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Knowledge base not found.", status_code=404
    )

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "some query"},
    )

    assert response.status_code == 404


async def test_search_returns_403_when_kb_belongs_to_another_user(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """Returns 403 when KB exists but belongs to a different user."""
    mock_kb_svc.get.side_effect = KnowledgeBaseServiceError(
        "Access to this knowledge base is forbidden.", status_code=403
    )

    response = await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "some query"},
    )

    assert response.status_code == 403


async def test_search_returns_401_when_unauthenticated(
    client_no_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """Returns 401 when no authentication is provided."""
    response = await client_no_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "some query"},
    )

    assert response.status_code == 401


async def test_search_passes_tracer_to_retrieval_service(
    client_auth, mock_kb_svc, mock_retrieval_svc, mock_db_session
):
    """RetrievalService.search() is called with a tracer kwarg for LangFuse tracing."""
    from baize.core.observability import TraceCollector

    mock_kb_svc.get.return_value = MagicMock(id=_KB_ID)
    mock_retrieval_svc.search = AsyncMock(return_value=[])
    mock_db_session.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[])))

    await client_auth.post(
        f"{_BASE_URL}/{_KB_ID}/search",
        json={"query": "trace test"},
    )

    call_kwargs = mock_retrieval_svc.search.call_args.kwargs
    assert "tracer" in call_kwargs
    assert isinstance(call_kwargs["tracer"], TraceCollector)
