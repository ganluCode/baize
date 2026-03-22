"""Tests for FastAPI app factory (F-003)."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


async def test_health_body_has_status_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    body = response.json()
    assert body["status"] == "ok"


async def test_health_no_auth_required(client: AsyncClient) -> None:
    """Health endpoint is accessible without any Authorization header."""
    response = await client.get("/health")
    assert response.status_code == 200


async def test_openapi_json_returns_200(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200


async def test_openapi_title(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.json()["info"]["title"] == "Baize API"


async def test_openapi_version(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.json()["info"]["version"] == "1.0.0"


async def test_docs_returns_200(client: AsyncClient) -> None:
    response = await client.get("/docs")
    assert response.status_code == 200


async def test_redoc_returns_200(client: AsyncClient) -> None:
    response = await client.get("/redoc")
    assert response.status_code == 200


async def test_openapi_contains_required_tags(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    schema = response.json()
    tag_names = {t["name"] for t in schema.get("tags", [])}
    required_tags = {"auth", "chat", "agents", "sessions", "tasks", "llm", "users"}
    assert required_tags.issubset(tag_names), f"Missing tags: {required_tags - tag_names}"


async def test_routes_use_api_v1_prefix(client: AsyncClient) -> None:
    """Verify that module routes are registered under /api/v1."""
    response = await client.get("/openapi.json")
    paths = set(response.json().get("paths", {}).keys())
    api_paths = {p for p in paths if p != "/health"}
    assert all(p.startswith("/api/v1") for p in api_paths), (
        f"Non-/api/v1 paths: {[p for p in api_paths if not p.startswith('/api/v1')]}"
    )
