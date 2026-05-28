"""ORM model tests for KnowledgeBase, KnowledgeDocument, KnowledgeChunk."""

from dotenv import load_dotenv

load_dotenv()

import asyncio  # noqa: E402
import os  # noqa: E402
import uuid  # noqa: E402

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

import baize.knowledge.models  # noqa: E402, F401 – registers knowledge models in Base.metadata
from baize.knowledge.models import KnowledgeBaseModel, KnowledgeChunkModel, KnowledgeDocumentModel  # noqa: E402
from baize.user.models import Base  # noqa: E402

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://baize:password@localhost:5432/baize_test"
)

# ---------------------------------------------------------------------------
# Schema bootstrap (once per module)
# ---------------------------------------------------------------------------


async def _ensure_schema() -> None:
    """Create all tables if they don't exist (idempotent)."""
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


@pytest.fixture(scope="module", autouse=True)
def module_schema():
    asyncio.run(_ensure_schema())
    yield


# ---------------------------------------------------------------------------
# Per-test DB fixture with targeted cleanup
# ---------------------------------------------------------------------------

_created_ids: dict[str, list[str]] = {
    "knowledge_chunks": [],
    "knowledge_documents": [],
    "knowledge_bases": [],
}


def _track(table: str, row_id: uuid.UUID) -> None:
    _created_ids[table].append(str(row_id))


@pytest_asyncio.fixture
async def db():
    """Yield an async DB session; DELETE only rows created during this test."""
    for key in _created_ids:
        _created_ids[key].clear()

    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

        # Children first to respect implicit ordering
        for table in ["knowledge_chunks", "knowledge_documents", "knowledge_bases"]:
            ids = _created_ids[table]
            if ids:
                placeholders = ", ".join(f"'{i}'" for i in ids)
                await session.execute(text(f"DELETE FROM {table} WHERE id IN ({placeholders})"))
        await session.commit()
    await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_create_knowledge_base(db: AsyncSession):
    kb = KnowledgeBaseModel(
        user_id=uuid.uuid4(),
        name="Test KB",
        embedding_provider="zhipu",
        embedding_model="embedding-3",
        embedding_dim=2048,
        status="active",
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    _track("knowledge_bases", kb.id)

    result = await db.get(KnowledgeBaseModel, kb.id)
    assert result is not None
    assert result.name == "Test KB"
    assert result.embedding_dim == 2048
    assert result.status == "active"


async def test_create_knowledge_document(db: AsyncSession):
    kb_id = uuid.uuid4()
    content_hash = "a" * 64
    doc = KnowledgeDocumentModel(
        kb_id=kb_id,
        title="Test Document",
        source_type="text",
        content_hash=content_hash,
        status="pending",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    _track("knowledge_documents", doc.id)

    result = await db.get(KnowledgeDocumentModel, doc.id)
    assert result is not None
    assert result.title == "Test Document"
    assert result.content_hash == content_hash
    assert result.status == "pending"


async def test_create_knowledge_chunk(db: AsyncSession):
    kb_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunk = KnowledgeChunkModel(
        kb_id=kb_id,
        doc_id=doc_id,
        level=0,
        content="This is a test chunk for full text search.",
    )
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    _track("knowledge_chunks", chunk.id)

    result = await db.get(KnowledgeChunkModel, chunk.id)
    assert result is not None
    assert result.content == "This is a test chunk for full text search."
    assert result.level == 0
    assert result.content_tsv is not None
