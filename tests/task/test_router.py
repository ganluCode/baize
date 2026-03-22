"""Integration tests for Task CRUD API endpoints (F-011).

Uses a dedicated `baize_test` PostgreSQL database so the test run never
touches the development database.  The schema is created once per module
and each test function gets its own engine + clean tables.
"""

import asyncio
import hashlib
import os
import uuid
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set env vars before any baize import
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.main import app as _app  # noqa: E402
from baize.core.database import get_db  # noqa: E402
from baize.core.deps import get_container  # noqa: E402
import baize.task.models  # noqa: F401 — registers Task in Base.metadata
from baize.user.models import Base, UserModel  # noqa: E402

# ---------------------------------------------------------------------------
# Test DB configuration
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "postgresql+asyncpg://baize:password@localhost:5432/baize_test"


# ---------------------------------------------------------------------------
# Async helpers for schema create / drop
# ---------------------------------------------------------------------------


async def _create_schema() -> None:
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def _drop_schema() -> None:
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Module-scoped sync fixture: schema lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def module_schema():
    """Create the test schema before this module; drop it after."""
    asyncio.run(_create_schema())
    yield
    asyncio.run(_drop_schema())


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db():
    """Yield a clean DB session; truncate task and user tables before each test."""
    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        await session.execute(
            text("TRUNCATE tasks, system_users RESTART IDENTITY CASCADE")
        )
        await session.commit()
        yield session
    await engine.dispose()


# ---------------------------------------------------------------------------
# Helpers — create seed data via ORM
# ---------------------------------------------------------------------------


async def _create_user(db: AsyncSession, api_key: str, name: str = "testuser") -> UserModel:
    """Insert a user with the given plaintext API key into the test DB."""
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    user = UserModel(
        email=f"{name}@example.com",
        name=name,
        password="hashed-not-used",
        role="user",
        api_key_hash=api_key_hash,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def api_client(db):
    """AsyncClient with test DB injected into the app."""
    engine = db.bind

    async def override_get_db():
        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        async with session_factory() as session:
            yield session

    mock_container = MagicMock()

    _app.dependency_overrides[get_db] = override_get_db
    _app.dependency_overrides[get_container] = lambda: mock_container

    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as client:
        yield client

    _app.dependency_overrides.pop(get_db, None)
    _app.dependency_overrides.pop(get_container, None)


# ---------------------------------------------------------------------------
# POST /api/v1/tasks
# ---------------------------------------------------------------------------


async def test_create_task_api(db, api_client):
    """POST /api/v1/tasks returns 201 with id, title, source=manual."""
    await _create_user(db, "key-create-001")

    response = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Write documentation"},
        headers={"X-API-Key": "key-create-001"},
    )

    assert response.status_code == 201
    body = response.json()
    assert "id" in body
    assert body["title"] == "Write documentation"
    assert body["source"] == "manual"


async def test_create_task_api_with_priority_and_due_date(db, api_client):
    """POST /api/v1/tasks accepts priority and due_date."""
    await _create_user(db, "key-create-002", name="createuser2")

    response = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Deploy service", "priority": "high", "due_date": "2026-03-25"},
        headers={"X-API-Key": "key-create-002"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["priority"] == "high"
    assert body["due_date"] == "2026-03-25"


async def test_unauthenticated_request_returns_401(db, api_client):
    """Any endpoint without API key returns 401."""
    response = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Should fail"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/tasks
# ---------------------------------------------------------------------------


async def test_list_tasks_api_returns_only_current_user_tasks(db, api_client):
    """GET /api/v1/tasks returns only the authenticated user's tasks."""
    user1 = await _create_user(db, "key-list-001", name="listuser1")
    user2 = await _create_user(db, "key-list-002", name="listuser2")

    # Create task for user1 via API
    await api_client.post(
        "/api/v1/tasks",
        json={"title": "User1 Task"},
        headers={"X-API-Key": "key-list-001"},
    )
    # Create task for user2 via API
    await api_client.post(
        "/api/v1/tasks",
        json={"title": "User2 Task"},
        headers={"X-API-Key": "key-list-002"},
    )

    response = await api_client.get(
        "/api/v1/tasks",
        headers={"X-API-Key": "key-list-001"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["title"] == "User1 Task"


async def test_list_tasks_api_with_filters(db, api_client):
    """GET /api/v1/tasks?status=todo returns only todo tasks."""
    await _create_user(db, "key-list-003", name="listuser3")

    # Create a todo task
    resp1 = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Todo Task"},
        headers={"X-API-Key": "key-list-003"},
    )
    task_id = resp1.json()["id"]

    # Mark it done
    await api_client.patch(
        f"/api/v1/tasks/{task_id}",
        json={"status": "done"},
        headers={"X-API-Key": "key-list-003"},
    )

    # Create another todo task
    await api_client.post(
        "/api/v1/tasks",
        json={"title": "Still Todo"},
        headers={"X-API-Key": "key-list-003"},
    )

    response = await api_client.get(
        "/api/v1/tasks?status=todo",
        headers={"X-API-Key": "key-list-003"},
    )

    assert response.status_code == 200
    body = response.json()
    titles = [item["title"] for item in body["items"]]
    assert "Still Todo" in titles
    assert "Todo Task" not in titles


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


async def test_get_task_returns_200(db, api_client):
    await _create_user(db, "key-get-001", name="getuser1")

    create_resp = await api_client.post(
        "/api/v1/tasks",
        json={"title": "My Task"},
        headers={"X-API-Key": "key-get-001"},
    )
    task_id = create_resp.json()["id"]

    response = await api_client.get(
        f"/api/v1/tasks/{task_id}",
        headers={"X-API-Key": "key-get-001"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == task_id


async def test_get_task_not_found(db, api_client):
    """GET /api/v1/tasks/{non_existent_id} returns 404."""
    await _create_user(db, "key-get-404", name="getuser404")

    response = await api_client.get(
        f"/api/v1/tasks/{uuid.uuid4()}",
        headers={"X-API-Key": "key-get-404"},
    )

    assert response.status_code == 404


async def test_get_task_forbidden(db, api_client):
    """GET /api/v1/tasks/{other_user_task_id} returns 403."""
    await _create_user(db, "key-get-owner", name="getowner")
    await _create_user(db, "key-get-other", name="getother")

    # Create task as owner
    create_resp = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Owner Task"},
        headers={"X-API-Key": "key-get-owner"},
    )
    task_id = create_resp.json()["id"]

    # Access as other user
    response = await api_client.get(
        f"/api/v1/tasks/{task_id}",
        headers={"X-API-Key": "key-get-other"},
    )

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# PATCH /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


async def test_patch_task_done_fills_completed_at(db, api_client):
    """PATCH status=done sets completed_at to a non-null value."""
    await _create_user(db, "key-patch-001", name="patchuser1")

    create_resp = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Task to complete"},
        headers={"X-API-Key": "key-patch-001"},
    )
    task_id = create_resp.json()["id"]

    response = await api_client.patch(
        f"/api/v1/tasks/{task_id}",
        json={"status": "done"},
        headers={"X-API-Key": "key-patch-001"},
    )

    assert response.status_code == 200
    assert response.json()["completed_at"] is not None
    assert response.json()["status"] == "done"


async def test_patch_task_reopen_clears_completed_at(db, api_client):
    """PATCH status=todo after done clears completed_at."""
    await _create_user(db, "key-patch-002", name="patchuser2")

    create_resp = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Reopenable Task"},
        headers={"X-API-Key": "key-patch-002"},
    )
    task_id = create_resp.json()["id"]

    # Mark done first
    await api_client.patch(
        f"/api/v1/tasks/{task_id}",
        json={"status": "done"},
        headers={"X-API-Key": "key-patch-002"},
    )

    # Reopen
    response = await api_client.patch(
        f"/api/v1/tasks/{task_id}",
        json={"status": "todo"},
        headers={"X-API-Key": "key-patch-002"},
    )

    assert response.status_code == 200
    assert response.json()["completed_at"] is None
    assert response.json()["status"] == "todo"


# ---------------------------------------------------------------------------
# DELETE /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


async def test_delete_task(db, api_client):
    """DELETE returns 204; subsequent GET returns 404."""
    await _create_user(db, "key-delete-001", name="deleteuser1")

    create_resp = await api_client.post(
        "/api/v1/tasks",
        json={"title": "Task to delete"},
        headers={"X-API-Key": "key-delete-001"},
    )
    task_id = create_resp.json()["id"]

    delete_resp = await api_client.delete(
        f"/api/v1/tasks/{task_id}",
        headers={"X-API-Key": "key-delete-001"},
    )
    assert delete_resp.status_code == 204

    get_resp = await api_client.get(
        f"/api/v1/tasks/{task_id}",
        headers={"X-API-Key": "key-delete-001"},
    )
    assert get_resp.status_code == 404
