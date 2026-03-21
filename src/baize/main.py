"""FastAPI application entry point for Baize."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from baize.core.config import Settings
from baize.core.container import Container
from baize.core.deps import set_container
from baize.llm.router import router as llm_router
from baize.user.auth_router import router as auth_router
from baize.user.router import router as user_router
from baize.user.seed import seed_admin_user

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize resources on startup, release on shutdown."""
    logger.info("Baize API starting up...")
    settings = Settings()
    container = Container(settings)
    set_container(container)
    await seed_admin_user()
    yield
    logger.info("Baize API shutting down...")
    await container.close()


app = FastAPI(title="Baize API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
