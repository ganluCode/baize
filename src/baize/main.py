"""FastAPI application entry point for Baize."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from baize.agent.chat_router import router as chat_router
from baize.agent.router import router as agent_router
from baize.core.config import Settings
from baize.core.container import Container
from baize.core.deps import set_container
from baize.core.exceptions import register_exception_handlers
from baize.core.logging import setup_logging
from baize.llm.router import router as llm_router
from baize.session.router import router as session_router
from baize.session.router import session_router as session_detail_router
from baize.task.router import router as task_router
from baize.user.auth_router import router as auth_router
from baize.user.router import router as user_router
from baize.user.seed import seed_admin_user

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize resources on startup, release on shutdown."""
    settings = Settings()
    setup_logging(settings.log_level)
    structlog.get_logger().info("application_startup", version="0.1.0")
    container = Container(settings)
    set_container(container)
    await seed_admin_user()
    yield
    logger.info("Baize API shutting down...")
    await container.close()


app = FastAPI(
    title="Baize API",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth"},
        {"name": "chat"},
        {"name": "agents"},
        {"name": "sessions"},
        {"name": "tasks"},
        {"name": "llm"},
        {"name": "users"},
    ],
    lifespan=lifespan,
)

register_exception_handlers(app)

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
app.include_router(agent_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(session_router, prefix="/api/v1")
app.include_router(session_detail_router, prefix="/api/v1")
app.include_router(task_router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
