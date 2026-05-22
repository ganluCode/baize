"""Integration tests for global exception handlers."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from baize.core.exceptions import register_exception_handlers


@pytest.fixture
def test_app() -> FastAPI:
    """Minimal FastAPI app with exception handlers registered."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-401")
    async def raise_401():
        raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/raise-403")
    async def raise_403():
        raise HTTPException(status_code=403, detail="Forbidden")

    @app.get("/raise-404")
    async def raise_404():
        raise HTTPException(status_code=404, detail="Not found")

    @app.get("/raise-400")
    async def raise_400():
        raise HTTPException(status_code=400, detail="Bad request")

    @app.get("/raise-500")
    async def raise_500():
        raise HTTPException(status_code=500, detail="Server error")

    @app.get("/raise-exception")
    async def raise_exception():
        raise RuntimeError("Something went terribly wrong")

    @app.post("/body-required")
    async def body_required(payload: dict):
        return payload

    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app, raise_server_exceptions=False)


def test_http_401_maps_to_error_code_40100(client: TestClient):
    response = client.get("/raise-401")
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 40100
    assert body["data"] is None
    assert isinstance(body["message"], str) and body["message"]


def test_http_403_maps_to_error_code_40300(client: TestClient):
    response = client.get("/raise-403")
    assert response.status_code == 403
    body = response.json()
    assert body["code"] == 40300
    assert body["data"] is None


def test_http_404_maps_to_error_code_40400(client: TestClient):
    response = client.get("/raise-404")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 40400
    assert body["data"] is None


def test_http_400_maps_to_error_code_40000(client: TestClient):
    response = client.get("/raise-400")
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 40000
    assert body["data"] is None


def test_validation_error_returns_400_with_code_40000(client: TestClient):
    """POST with a non-dict body triggers RequestValidationError."""
    response = client.post("/body-required", content="not-json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 40000
    assert body["data"] is None
    # message should describe the field error
    assert body["message"]


def test_unhandled_exception_returns_500_with_code_50000(client: TestClient):
    response = client.get("/raise-exception")
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == 50000
    assert body["data"] is None
    # Internal stack trace must NOT be exposed
    assert "terribly wrong" not in body["message"]
    assert "Traceback" not in body["message"]


def test_register_exception_handlers_accepts_fastapi_instance():
    """register_exception_handlers should not raise when given a FastAPI app."""
    app = FastAPI()
    register_exception_handlers(app)  # must not raise
