"""Knowledge API test infrastructure (F-010).

Fixtures:
- fake_redis:                 fakeredis.aioredis.FakeRedis — no real Redis
- test_jwt_service:           JWTService backed by fake_redis
- mock_db_session:            Per-test isolated AsyncSession mock
- mock_qdrant:                Mock AsyncQdrantClient
- mock_container:             Mock Container with test JWT service
- user_registry:              dict[token -> UserModel] for in-test auth
- knowledge_app:              FastAPI app wired for knowledge API integration tests
- api_client:                 AsyncClient (unauthenticated by default)
- create_test_user_and_token: Factory → (user_id: UUID, token: str)

Module-level utility:
- wait_for_document_status:   Polls document status; raises AssertionError on timeout
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from fastapi import Header, HTTPException
from httpx import ASGITransport, AsyncClient

from baize.user.jwt_service import JWTService
from baize.user.models import UserModel


# ─── Redis & JWT ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    """In-memory async Redis via fakeredis — no real Redis server needed."""
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def test_jwt_service(fake_redis: fakeredis.aioredis.FakeRedis) -> JWTService:
    """JWTService backed by fakeredis for issuing and verifying test tokens."""
    return JWTService(secret_key="test-jwt-secret-key-for-unit-tests", redis=fake_redis)


# ─── DB session ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Per-test isolated AsyncSession mock.

    Each test gets a fresh object — no shared state between tests.
    Default execute() returns empty results; configure per-test as needed.
    """
    session = AsyncMock()

    _empty = MagicMock()
    _empty.scalar_one_or_none.return_value = None
    _empty.scalar_one.return_value = 0
    _empty.one_or_none.return_value = None
    _scalars = MagicMock()
    _scalars.all.return_value = []
    _empty.scalars.return_value = _scalars
    _empty.all.return_value = []

    session.execute = AsyncMock(return_value=_empty)
    session.get = AsyncMock(return_value=None)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.refresh = AsyncMock()
    return session


# ─── Qdrant ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Mock AsyncQdrantClient — no external Qdrant service required."""
    mock = MagicMock()
    mock.upsert = AsyncMock()
    mock.delete = AsyncMock()
    mock.query_points = AsyncMock(return_value=MagicMock(points=[]))
    mock.search = AsyncMock(return_value=[])
    mock.create_collection = AsyncMock()
    mock.collection_exists = AsyncMock(return_value=True)
    mock.get_collection = AsyncMock()
    mock.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    return mock


# ─── Container ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_container(test_jwt_service: JWTService) -> MagicMock:
    """Mock Container with test JWTService and no-op placeholders for other services."""
    container = MagicMock()
    container.jwt_service = test_jwt_service
    container.provider_factory = MagicMock()
    container.model_router = MagicMock()
    container.memory_service = None
    return container


# ─── User registry (token → user) ────────────────────────────────────────────


@pytest.fixture
def user_registry() -> dict:
    """Maps auth tokens to UserModel mocks for test-time authentication.

    Populated by create_test_user_and_token; consumed by knowledge_app's
    auth override.
    """
    return {}


# ─── knowledge_app ───────────────────────────────────────────────────────────


@pytest.fixture
def knowledge_app(
    app,
    mock_db_session: AsyncMock,
    mock_qdrant: MagicMock,
    mock_container: MagicMock,
    user_registry: dict,
):
    """FastAPI app configured for knowledge API integration tests.

    Overrides applied:
    - get_db              → yields mock_db_session (isolated per test)
    - get_container       → returns mock_container (with test JWT service)
    - get_current_user    → looks up Bearer/X-API-Key token in user_registry
    - Qdrant _client      → replaced with mock_qdrant (singleton bypass)
    - AsyncSessionLocal   → async CM yielding mock_db_session (for background tasks)
    - core deps _container → mock_container (for background task get_container() calls)

    BackgroundTasks run synchronously because AsyncClient + ASGITransport awaits
    the full ASGI lifecycle including background tasks before returning.
    """
    from baize.core.database import get_db
    from baize.core.deps import get_container
    from baize.user.deps import get_current_user
    import baize.knowledge.ingestion.qdrant_client as _qdrant_mod
    import baize.core.deps as _deps_mod

    async def _db_override() -> AsyncGenerator:
        yield mock_db_session

    async def _auth_override(
        authorization: str | None = Header(default=None, alias="Authorization"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> UserModel:
        token: str | None = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
        if token is None and x_api_key:
            token = x_api_key
        if token and token in user_registry:
            return user_registry[token]
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Async context manager used by background tasks via `async with AsyncSessionLocal()`
    class _SessionCM:
        async def __aenter__(self) -> AsyncMock:
            return mock_db_session

        async def __aexit__(self, *args: object) -> None:
            pass

    class _MockSessionFactory:
        def __call__(self) -> _SessionCM:
            return _SessionCM()

    _orig_qdrant = _qdrant_mod._client
    _orig_container = _deps_mod._container

    _qdrant_mod._client = mock_qdrant
    _deps_mod._container = mock_container

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[get_container] = lambda: mock_container
    app.dependency_overrides[get_current_user] = _auth_override

    with patch("baize.core.database.AsyncSessionLocal", new=_MockSessionFactory()):
        yield app

    # Restore everything after the test
    _qdrant_mod._client = _orig_qdrant
    _deps_mod._container = _orig_container
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_container, None)
    app.dependency_overrides.pop(get_current_user, None)


# ─── AsyncClient ─────────────────────────────────────────────────────────────


@pytest.fixture
async def api_client(knowledge_app) -> AsyncGenerator[AsyncClient, None]:
    """Unauthenticated async HTTP client for knowledge API tests.

    Add an Authorization header per-request or use create_test_user_and_token
    to obtain a token.
    """
    async with AsyncClient(
        transport=ASGITransport(app=knowledge_app), base_url="http://test"
    ) as client:
        yield client


# ─── create_test_user_and_token ──────────────────────────────────────────────


@pytest.fixture
def create_test_user_and_token(user_registry: dict):
    """Factory fixture: create a mock user and register an auth token for it.

    Returns a callable with signature:
        (username: str | None = None, role: str = "user") -> (user_id: UUID, token: str)

    The returned token can be passed directly as a Bearer token:
        headers={"Authorization": f"Bearer {token}"}

    Each call creates a new user with a unique ID and token.
    """

    def _factory(
        username: str | None = None,
        role: str = "user",
    ) -> tuple[uuid.UUID, str]:
        uid = uuid.uuid4()
        uname = username or f"user_{uid.hex[:8]}"
        token = f"test-bearer-{uid.hex}"

        user = MagicMock(spec=UserModel)
        user.id = uid
        user.name = uname
        user.email = f"{uname}@test.example"
        user.role = role
        user.is_active = True

        user_registry[token] = user
        return uid, token

    return _factory


# ─── wait_for_document_status ────────────────────────────────────────────────


async def wait_for_document_status(
    client: AsyncClient,
    kb_id: str | uuid.UUID,
    doc_id: str | uuid.UUID,
    expected_status: str,
    token: str,
    *,
    timeout: int = 10,
) -> dict:
    """Poll GET /{kb_id}/documents/{doc_id} until status matches expected_status.

    With AsyncClient + ASGITransport, background tasks execute synchronously
    before the triggering request returns, so this typically resolves on the
    first poll.  The timeout is a safety net for any edge cases.

    Args:
        client:          AsyncClient connected to the test app.
        kb_id:           Knowledge base UUID (string or UUID).
        doc_id:          Document UUID (string or UUID).
        expected_status: Status value to wait for (e.g. 'ingested', 'failed').
        token:           Bearer token for the Authorization header.
        timeout:         Maximum seconds to wait before raising AssertionError.

    Returns:
        Document JSON dict when expected_status is reached.

    Raises:
        AssertionError: If expected_status is not reached within timeout seconds.
    """
    import time

    url = f"/api/v1/knowledge-bases/{kb_id}/documents/{doc_id}"
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + timeout
    last_response = None

    while time.monotonic() < deadline:
        last_response = await client.get(url, headers=headers)
        if last_response.status_code == 200:
            data = last_response.json()
            if data.get("status") == expected_status:
                return data
        await asyncio.sleep(0.05)

    if last_response is not None and last_response.status_code == 200:
        last_status = last_response.json().get("status", "<missing>")
    else:
        last_status = last_response.status_code if last_response is not None else "no_response"

    raise AssertionError(
        f"Document {doc_id} did not reach status={expected_status!r} within "
        f"{timeout}s (last observed: {last_status!r})"
    )
