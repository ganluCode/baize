"""Database engine and session management for Baize."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from baize.core.config import Settings

_settings = Settings()

engine = create_async_engine(
    _settings.database_url,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for use with FastAPI Depends."""
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()
