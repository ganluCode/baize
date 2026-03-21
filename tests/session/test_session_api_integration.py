"""Integration tests for Session and Messages API endpoints (F-012).

Uses a dedicated `baize_test` PostgreSQL database so the test run never
touches the development database.  The schema is created once per module
and each test function gets its own engine + clean tables.
"""

import asyncio
import hashlib
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Table, Column, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set env vars before any baize import
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.main import app as _app  # noqa: E402
from baize.core.database import get_db  # noqa: E402
from baize.core.deps import get_container  # noqa: E402
from baize.session.models import ChatMessageModel, MessageRole, SessionModel  # noqa: E402
from baize.user.models import Base, UserModel  # noqa: E402

# ---------------------------------------------------------------------------
# Test DB configuration
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "postgresql+asyncpg://baize:password@localhost:5432/baize_test"

# Register the agent_configs stub table so Base.metadata.create_all creates it.
_agent_configs_table = Table(
    "agent_configs",
    Base.metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    extend_existing=True,
)


# ---------------------------------------------------------------------------
# Async helpers for schema create / drop (called from sync fixtures)
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
# Function-scoped fixtures — one engine per test avoids event-loop conflicts
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db():
    """Yield a clean DB session; truncate all tables before each test."""
    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        await session.execute(
            text(
                "TRUNCATE chat_messages, sessions, system_users, agent_configs"
                " RESTART IDENTITY CASCADE"
            )
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


async def _create_agent(db: AsyncSession) -> uuid.UUID:
    """Insert a minimal agent_configs row and return its id."""
    agent_id = uuid.uuid4()
    await db.execute(
        text("INSERT INTO agent_configs (id) VALUES (:id)"),
        {"id": str(agent_id)},
    )
    await db.commit()
    return agent_id


async def _create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID,
    title: str | None = None,
    status: str = "active",
    updated_at: datetime | None = None,
) -> SessionModel:
    """Insert a session row."""
    from baize.session.models import SessionStatus

    session = SessionModel(
        user_id=user_id,
        agent_id=agent_id,
        title=title,
        status=SessionStatus(status),
        updated_at=updated_at or datetime.now(timezone.utc),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def _create_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    role: str = "user",
) -> ChatMessageModel:
    """Insert a chat message."""
    msg = ChatMessageModel(
        session_id=session_id,
        user_id=user_id,
        role=MessageRole(role),
        content=content,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


# ---------------------------------------------------------------------------
# App fixture — overrides get_db + get_container per test
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def api_client(db):
    """AsyncClient with test DB injected into the app."""
    engine = db.bind

    async def override_get_db():
        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        async with session_factory() as session:
            yield session

    # Minimal mock container — jwt_service only matters for Bearer tokens
    # (our integration tests use X-API-Key auth only)
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
# POST /api/v1/agents/{agent_id}/sessions
# ---------------------------------------------------------------------------


async def test_create_session_returns_201(db, api_client):
    user = await _create_user(db, "valid-key-001")
    agent_id = await _create_agent(db)

    response = await api_client.post(
        f"/api/v1/agents/{agent_id}/sessions",
        json={"title": "Hello"},
        headers={"X-API-Key": "valid-key-001"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Hello"
    assert body["status"] == "active"
    assert body["agent_id"] == str(agent_id)
    assert body["user_id"] == str(user.id)


async def test_create_session_without_title_returns_null_title(db, api_client):
    await _create_user(db, "valid-key-002")
    agent_id = await _create_agent(db)

    response = await api_client.post(
        f"/api/v1/agents/{agent_id}/sessions",
        json={},
        headers={"X-API-Key": "valid-key-002"},
    )

    assert response.status_code == 201
    assert response.json()["title"] is None


async def test_create_session_invalid_api_key_returns_401(db, api_client):
    agent_id = await _create_agent(db)

    response = await api_client.post(
        f"/api/v1/agents/{agent_id}/sessions",
        json={},
        headers={"X-API-Key": "this-key-does-not-exist"},
    )

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/agents/{agent_id}/sessions
# ---------------------------------------------------------------------------


async def test_list_sessions_returns_only_current_user_data(db, api_client):
    """Two users exist; GET returns only the authenticated user's sessions."""
    user1 = await _create_user(db, "key-user1", name="user1")
    user2 = await _create_user(db, "key-user2", name="user2")
    agent_id = await _create_agent(db)

    await _create_session(db, user1.id, agent_id, title="User1 Session")
    await _create_session(db, user2.id, agent_id, title="User2 Session")

    response = await api_client.get(
        f"/api/v1/agents/{agent_id}/sessions",
        headers={"X-API-Key": "key-user1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["title"] == "User1 Session"


async def test_list_sessions_status_filter(db, api_client):
    """GET with status=active excludes archived sessions."""
    user = await _create_user(db, "key-filter-status")
    agent_id = await _create_agent(db)

    await _create_session(db, user.id, agent_id, title="Active", status="active")
    await _create_session(db, user.id, agent_id, title="Archived", status="archived")

    response = await api_client.get(
        f"/api/v1/agents/{agent_id}/sessions?status=active",
        headers={"X-API-Key": "key-filter-status"},
    )

    assert response.status_code == 200
    body = response.json()
    titles = [item["title"] for item in body["items"]]
    assert "Active" in titles
    assert "Archived" not in titles


async def test_list_sessions_pagination(db, api_client):
    """limit and offset parameters work correctly."""
    user = await _create_user(db, "key-pagination")
    agent_id = await _create_agent(db)

    for i in range(5):
        await _create_session(db, user.id, agent_id, title=f"Session {i}")

    response = await api_client.get(
        f"/api/v1/agents/{agent_id}/sessions?limit=2&offset=0",
        headers={"X-API-Key": "key-pagination"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert len(body["items"]) == 2


async def test_list_sessions_ordered_by_updated_at_desc(db, api_client):
    """Sessions are returned newest-first (updated_at descending)."""
    user = await _create_user(db, "key-ordering")
    agent_id = await _create_agent(db)

    older_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    newer_time = datetime(2025, 6, 1, tzinfo=timezone.utc)

    await _create_session(db, user.id, agent_id, title="Older", updated_at=older_time)
    await _create_session(db, user.id, agent_id, title="Newer", updated_at=newer_time)

    response = await api_client.get(
        f"/api/v1/agents/{agent_id}/sessions",
        headers={"X-API-Key": "key-ordering"},
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["title"] == "Newer"
    assert items[1]["title"] == "Older"


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_get_session_returns_200(db, api_client):
    user = await _create_user(db, "key-get-200")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user.id, agent_id, title="My Session")

    response = await api_client.get(
        f"/api/v1/sessions/{session.id}",
        headers={"X-API-Key": "key-get-200"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(session.id)


async def test_get_session_not_found_returns_404(db, api_client):
    await _create_user(db, "key-get-404")

    response = await api_client.get(
        f"/api/v1/sessions/{uuid.uuid4()}",
        headers={"X-API-Key": "key-get-404"},
    )

    assert response.status_code == 404


async def test_get_session_wrong_owner_returns_403(db, api_client):
    user1 = await _create_user(db, "key-owner-1", name="owner1")
    await _create_user(db, "key-owner-2", name="owner2")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user1.id, agent_id)

    response = await api_client.get(
        f"/api/v1/sessions/{session.id}",
        headers={"X-API-Key": "key-owner-2"},
    )

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# PATCH /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_archive_session_excluded_from_active_list(db, api_client):
    """After archiving, the session no longer appears in the active-only list."""
    user = await _create_user(db, "key-archive")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user.id, agent_id, title="To Archive")

    # Archive it
    patch_resp = await api_client.patch(
        f"/api/v1/sessions/{session.id}",
        json={"status": "archived"},
        headers={"X-API-Key": "key-archive"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "archived"

    # Should not appear in active-only list
    list_resp = await api_client.get(
        f"/api/v1/agents/{agent_id}/sessions?status=active",
        headers={"X-API-Key": "key-archive"},
    )
    assert list_resp.status_code == 200
    titles = [item["title"] for item in list_resp.json()["items"]]
    assert "To Archive" not in titles


async def test_archive_session_wrong_owner_returns_403(db, api_client):
    user1 = await _create_user(db, "key-arch-owner1", name="archowner1")
    await _create_user(db, "key-arch-owner2", name="archowner2")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user1.id, agent_id)

    response = await api_client.patch(
        f"/api/v1/sessions/{session.id}",
        json={"status": "archived"},
        headers={"X-API-Key": "key-arch-owner2"},
    )

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# DELETE /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_delete_session_returns_404_on_subsequent_get(db, api_client):
    """After DELETE, GET the session returns 404."""
    user = await _create_user(db, "key-delete")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user.id, agent_id)

    delete_resp = await api_client.delete(
        f"/api/v1/sessions/{session.id}",
        headers={"X-API-Key": "key-delete"},
    )
    assert delete_resp.status_code == 204

    get_resp = await api_client.get(
        f"/api/v1/sessions/{session.id}",
        headers={"X-API-Key": "key-delete"},
    )
    assert get_resp.status_code == 404


async def test_delete_session_messages_also_return_404(db, api_client):
    """After DELETE, GET messages for that session returns 404."""
    user = await _create_user(db, "key-delete-msgs")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user.id, agent_id)
    await _create_message(db, session.id, user.id, "hello")

    await api_client.delete(
        f"/api/v1/sessions/{session.id}",
        headers={"X-API-Key": "key-delete-msgs"},
    )

    msgs_resp = await api_client.get(
        f"/api/v1/sessions/{session.id}/messages",
        headers={"X-API-Key": "key-delete-msgs"},
    )
    assert msgs_resp.status_code == 404


async def test_delete_session_wrong_owner_returns_403(db, api_client):
    user1 = await _create_user(db, "key-del-owner1", name="delowner1")
    await _create_user(db, "key-del-owner2", name="delowner2")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user1.id, agent_id)

    response = await api_client.delete(
        f"/api/v1/sessions/{session.id}",
        headers={"X-API-Key": "key-del-owner2"},
    )

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}/messages
# ---------------------------------------------------------------------------


async def test_list_messages_ordered_by_created_at_asc(db, api_client):
    """Messages are returned in ascending order of created_at."""
    user = await _create_user(db, "key-msgs-order")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user.id, agent_id)

    # Insert messages with a small delay to ensure ordering
    await _create_message(db, session.id, user.id, "first")
    await asyncio.sleep(0.01)
    await _create_message(db, session.id, user.id, "second")
    await asyncio.sleep(0.01)
    await _create_message(db, session.id, user.id, "third")

    response = await api_client.get(
        f"/api/v1/sessions/{session.id}/messages",
        headers={"X-API-Key": "key-msgs-order"},
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    contents = [m["content"] for m in items]
    assert contents == ["first", "second", "third"]


async def test_list_messages_pagination(db, api_client):
    """limit and offset parameters restrict the returned messages."""
    user = await _create_user(db, "key-msgs-pages")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user.id, agent_id)

    for i in range(6):
        await _create_message(db, session.id, user.id, f"msg {i}")

    response = await api_client.get(
        f"/api/v1/sessions/{session.id}/messages?limit=3&offset=2",
        headers={"X-API-Key": "key-msgs-pages"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 6
    assert len(body["items"]) == 3


async def test_list_messages_total_reflects_full_count(db, api_client):
    """total field counts all messages even when paginated."""
    user = await _create_user(db, "key-msgs-total")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user.id, agent_id)

    for i in range(4):
        await _create_message(db, session.id, user.id, f"msg {i}")

    response = await api_client.get(
        f"/api/v1/sessions/{session.id}/messages?limit=2",
        headers={"X-API-Key": "key-msgs-total"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 4


async def test_list_messages_session_not_found_returns_404(db, api_client):
    await _create_user(db, "key-msgs-404")

    response = await api_client.get(
        f"/api/v1/sessions/{uuid.uuid4()}/messages",
        headers={"X-API-Key": "key-msgs-404"},
    )

    assert response.status_code == 404


async def test_list_messages_wrong_owner_returns_403(db, api_client):
    user1 = await _create_user(db, "key-msgs-owner1", name="msgsowner1")
    await _create_user(db, "key-msgs-owner2", name="msgsowner2")
    agent_id = await _create_agent(db)
    session = await _create_session(db, user1.id, agent_id)

    response = await api_client.get(
        f"/api/v1/sessions/{session.id}/messages",
        headers={"X-API-Key": "key-msgs-owner2"},
    )

    assert response.status_code == 403
