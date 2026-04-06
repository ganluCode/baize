"""GET /v1/models — list available agents as OpenAI model objects."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.models import AgentConfig
from baize.core.database import get_db
from baize.openai_compat.schemas import ModelListResponse, ModelObject
from baize.user.deps import get_current_user
from baize.user.models import UserModel

router = APIRouter(tags=["openai-compat"])


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ModelListResponse:
    """Return all enabled agents as OpenAI-compatible model objects."""
    result = await db.execute(
        select(AgentConfig)
        .where(
            AgentConfig.user_id == current_user.id,
            AgentConfig.is_enabled.is_(True),
        )
        .order_by(AgentConfig.name)
    )
    agents = result.scalars().all()

    data = [
        ModelObject(
            id=f"baize-{a.name}",
            created=int(a.created_at.timestamp()) if a.created_at else 0,
            owned_by="baize",
        )
        for a in agents
    ]
    return ModelListResponse(data=data)
