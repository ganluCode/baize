"""Integration tests for the authentication chain (F-011).

Covers three token types (JWT, API Key, Service Key), missing credentials,
invalid credentials, and service-key access to user-context-required endpoints.
Uses httpx AsyncClient against the real FastAPI app with mocked infrastructure
(no real Redis or PostgreSQL connections).
"""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.auth.schemas import AuthResult  # noqa: E402
from baize.core.database import get_db  # noqa: E402
from baize.core.deps import get_container, get_provider_factory  # noqa: E402
from baize.main import app as _app  # noqa: E402
from baize.user.models import UserModel  # noqa: E402
from baize.user.router import _get_user_service  # noqa: E402
from baize.user.schemas import UserResponse  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_ID = str(uuid.uuid4())
_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_TEST_SERVICE_KEY = "test-integration-service-key-001"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_model() -> MagicMock:
    user = MagicMock(spec=UserModel)
    user.id = uuid.UUID(_USER_ID)
    user.email = "test@example.com"
    user.name = "testuser"
    user.role = "user"
    user.is_active = True
    user.avatar = None
    user.preferences = None
    user.api_key_hash = hashlib.sha256(b"test-api-key-12345").hexdigest()
    user.created_at = _NOW
    user.updated_at = _NOW
    return user


def _make_user_response(user: MagicMock) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        avatar=user.avatar,
        preferences=user.preferences,
        is_active=user.is_active,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_container() -> MagicMock:
    container = MagicMock()
    container.config.secret_key = "test-secret-key"
    container.redis = AsyncMock()
    container.jwt_service.verify_token = AsyncMock(return_value={"user_id": _USER_ID})
    return container


def _make_provider_factory() -> MagicMock:
    factory = MagicMock()
    factory.list_providers.return_value = []
    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user():
    return _make_user_model()


@pytest.fixture
def mock_container():
    return _make_container()


# ---------------------------------------------------------------------------
# Tests 1 & 2 — legacy user.deps auth chain → GET /api/v1/users/me
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_jwt_on_users_me_returns_200(mock_user, mock_container):
    """Valid JWT in Authorization header → GET /api/v1/users/me → HTTP 200."""
    user_response = _make_user_response(mock_user)
    mock_svc = AsyncMock()
    mock_svc.get_me = AsyncMock(return_value=user_response)

    async def override_get_db():
        yield AsyncMock()

    _app.dependency_overrides[get_db] = override_get_db
    _app.dependency_overrides[get_container] = lambda: mock_container
    _app.dependency_overrides[_get_user_service] = lambda: mock_svc

    try:
        with patch(
            "baize.user.repository.UserRepository.get_by_id",
            new=AsyncMock(return_value=mock_user),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=_app), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/v1/users/me",
                    headers={"Authorization": "Bearer some-valid-jwt-token"},
                )
    finally:
        _app.dependency_overrides.pop(get_db, None)
        _app.dependency_overrides.pop(get_container, None)
        _app.dependency_overrides.pop(_get_user_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "test@example.com"
    assert body["name"] == "testuser"


@pytest.mark.asyncio
async def test_valid_api_key_on_users_me_returns_200(mock_user, mock_container):
    """Valid API key in X-API-Key header → GET /api/v1/users/me → HTTP 200."""
    user_response = _make_user_response(mock_user)
    mock_svc = AsyncMock()
    mock_svc.get_me = AsyncMock(return_value=user_response)

    async def override_get_db():
        yield AsyncMock()

    _app.dependency_overrides[get_db] = override_get_db
    _app.dependency_overrides[get_container] = lambda: mock_container
    _app.dependency_overrides[_get_user_service] = lambda: mock_svc

    try:
        with patch(
            "baize.user.repository.UserRepository.get_by_api_key_hash",
            new=AsyncMock(return_value=mock_user),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=_app), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/v1/users/me",
                    headers={"X-API-Key": "test-api-key-12345"},
                )
    finally:
        _app.dependency_overrides.pop(get_db, None)
        _app.dependency_overrides.pop(get_container, None)
        _app.dependency_overrides.pop(_get_user_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "test@example.com"


# ---------------------------------------------------------------------------
# Tests 3-6 — new auth.deps chain (Service Key → JWT → API Key)
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_chain_client(mock_container):
    """Return an async context manager that yields an AsyncClient.

    Overrides get_db, get_container, and get_provider_factory for tests that
    exercise the baize.auth.deps authentication chain.
    """

    class _ClientContext:
        async def __aenter__(self):
            async def override_get_db():
                yield AsyncMock()

            _app.dependency_overrides[get_db] = override_get_db
            _app.dependency_overrides[get_container] = lambda: mock_container
            _app.dependency_overrides[get_provider_factory] = _make_provider_factory
            self._client = AsyncClient(
                transport=ASGITransport(app=_app), base_url="http://test"
            )
            return await self._client.__aenter__()

        async def __aexit__(self, *args):
            _app.dependency_overrides.pop(get_db, None)
            _app.dependency_overrides.pop(get_container, None)
            _app.dependency_overrides.pop(get_provider_factory, None)
            return await self._client.__aexit__(*args)

    return _ClientContext()


@pytest.mark.asyncio
async def test_service_key_on_llm_providers_returns_200(auth_chain_client):
    """Service Key bearer token → GET /api/v1/llm/providers → HTTP 200."""
    service_result = AuthResult(user_id=None, auth_type="service")

    with patch("baize.auth.deps.authenticate_service_key", return_value=service_result):
        async with auth_chain_client as client:
            response = await client.get(
                "/api/v1/llm/providers",
                headers={"Authorization": f"Bearer {_TEST_SERVICE_KEY}"},
            )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_service_key_on_user_context_endpoint_returns_403(auth_chain_client):
    """Service Key on an endpoint requiring user context → HTTP 403, code=40300.

    POST /api/v1/chat uses baize.auth.deps.get_current_user which raises 403
    when a service-level auth result (user_id=None) is received.
    """
    service_result = AuthResult(user_id=None, auth_type="service")

    with patch("baize.auth.deps.authenticate_service_key", return_value=service_result):
        async with auth_chain_client as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "hello"},
                headers={"Authorization": f"Bearer {_TEST_SERVICE_KEY}"},
            )

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 40300


@pytest.mark.asyncio
async def test_missing_auth_header_returns_401_missing_credentials(auth_chain_client):
    """No Authorization header → HTTP 401, code=40100, message='Missing credentials'."""
    async with auth_chain_client as client:
        response = await client.get("/api/v1/llm/providers")

    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 40100
    assert body["message"] == "Missing credentials"


@pytest.mark.asyncio
async def test_invalid_token_returns_401_invalid_credentials(auth_chain_client):
    """Token rejected by all authenticators → HTTP 401, code=40100, message='Invalid credentials'."""
    with (
        patch("baize.auth.deps.authenticate_service_key", return_value=None),
        patch(
            "baize.auth.deps.authenticate_jwt",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "baize.auth.deps.authenticate_api_key",
            new=AsyncMock(return_value=None),
        ),
    ):
        async with auth_chain_client as client:
            response = await client.get(
                "/api/v1/llm/providers",
                headers={"Authorization": "Bearer totally-invalid-token"},
            )

    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 40100
    assert body["message"] == "Invalid credentials"
