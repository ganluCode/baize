"""FastAPI application entry point for Baize."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from baize.core.config import Settings
from baize.core.container import Container
from baize.core.deps import set_container

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize resources on startup, release on shutdown."""
    logger.info("Baize API starting up...")
    settings = Settings()
    container = Container(settings)
    set_container(container)
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


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
