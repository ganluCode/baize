"""Shared helpers for OpenAI-compatible routers."""

from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.models import AgentConfig


async def resolve_agent_by_model(
    model_name: str,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> AgentConfig:
    """Resolve an AgentConfig from an OpenAI-style model name.

    Tries exact match on ``model_name``, then strips ``baize-`` prefix.
    Falls back to the user's default agent if model_name is empty or "default".

    Raises:
        HTTPException: 404 if no matching agent found.
    """
    # Normalize: strip "baize-" prefix if present
    name = model_name
    if name.startswith("baize-"):
        name = name[len("baize-"):]

    if name and name != "default":
        # Try exact name match for this user
        result = await db.execute(
            select(AgentConfig).where(
                AgentConfig.user_id == user_id,
                AgentConfig.name == name,
                AgentConfig.is_enabled.is_(True),
            )
        )
        agent = result.scalar_one_or_none()
        if agent is not None:
            return agent

        # Try by UUID (agent_id passed as model name)
        try:
            agent_uuid = uuid.UUID(model_name)
            result = await db.execute(
                select(AgentConfig).where(
                    AgentConfig.id == agent_uuid,
                    AgentConfig.user_id == user_id,
                )
            )
            agent = result.scalar_one_or_none()
            if agent is not None:
                return agent
        except ValueError:
            pass

    # Fallback: user's default agent
    from baize.user.models import UserModel

    user_result = await db.execute(
        select(UserModel.default_agent_id).where(UserModel.id == user_id)
    )
    default_id = user_result.scalar_one_or_none()
    if default_id is not None:
        result = await db.execute(
            select(AgentConfig).where(AgentConfig.id == default_id)
        )
        agent = result.scalar_one_or_none()
        if agent is not None:
            return agent

    raise HTTPException(status_code=404, detail=f"Agent not found for model: {model_name}")
