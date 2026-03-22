"""Tests for src/baize/main.py (F-005)."""

import pytest
from httpx import ASGITransport, AsyncClient

from baize.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


async def test_health_response_body(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.json()["status"] == "ok"
