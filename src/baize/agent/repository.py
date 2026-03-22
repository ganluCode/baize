"""Repository for AgentConfig database operations."""

import uuid
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.models import AgentConfig
from baize.agent.schemas import AgentCreate, AgentUpdate


class AgentConfigRepository:
    """Data access layer for AgentConfig."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: uuid.UUID, data: AgentCreate) -> AgentConfig:
        """Insert a new agent config and return the persisted AgentConfig.

        Args:
            user_id: The owner of the new agent.
            data: Validated creation payload.

        Returns:
            The persisted AgentConfig.
        """
        obj = AgentConfig(
            user_id=user_id,
            name=data.name,
            description=data.description,
            system_prompt=data.system_prompt,
            tools=data.tools,
            model_config=data.llm_config,
            auto_memory_recall=data.auto_memory_recall,
            shared_memory=data.shared_memory,
        )
        self._session.add(obj)
        await self._session.commit()
        await self._session.refresh(obj)
        return obj

    async def get_by_id(self, agent_id: uuid.UUID) -> Optional[AgentConfig]:
        """Return the agent config with the given id, or None if not found.

        Args:
            agent_id: The agent config id to look up.

        Returns:
            The matching AgentConfig or None.
        """
        result = await self._session.execute(
            select(AgentConfig).where(AgentConfig.id == agent_id)
        )
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: uuid.UUID) -> list[AgentConfig]:
        """Return all agent configs belonging to a user.

        Args:
            user_id: Filter to agents owned by this user.

        Returns:
            A list of AgentConfig ordered by created_at ascending.
        """
        result = await self._session.execute(
            select(AgentConfig)
            .where(AgentConfig.user_id == user_id)
            .order_by(AgentConfig.created_at.asc())
        )
        return list(result.scalars().all())

    async def update(self, agent_id: uuid.UUID, data: AgentUpdate) -> Optional[AgentConfig]:
        """Apply a partial update to an agent config and return the refreshed record.

        Args:
            agent_id: The id of the agent config to update.
            data: Partial update payload; None fields are ignored.

        Returns:
            The updated AgentConfig, or None if not found.
        """
        obj = await self.get_by_id(agent_id)
        if obj is None:
            return None

        update_data = data.model_dump(exclude_none=True, by_alias=False)
        # llm_config maps to the model's model_config field
        if "llm_config" in update_data:
            update_data["model_config"] = update_data.pop("llm_config")

        for key, value in update_data.items():
            setattr(obj, key, value)

        await self._session.commit()
        await self._session.refresh(obj)
        return obj

    async def delete(self, agent_id: uuid.UUID) -> bool:
        """Delete an agent config.

        Args:
            agent_id: The id of the agent config to delete.

        Returns:
            True if the record was found and deleted, False otherwise.
        """
        obj = await self.get_by_id(agent_id)
        if obj is None:
            return False

        await self._session.delete(obj)
        await self._session.commit()
        return True

    async def get_default_by_user(self, user_id: uuid.UUID) -> Optional[AgentConfig]:
        """Return the default agent config for a user, or None.

        Args:
            user_id: The user whose default agent to look up.

        Returns:
            The AgentConfig with is_default=True, or None.
        """
        result = await self._session.execute(
            select(AgentConfig).where(
                AgentConfig.user_id == user_id,
                AgentConfig.is_default.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def set_default(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> None:
        """Set a specific agent as the default for a user.

        Clears is_default on all of the user's agents, then sets is_default=True
        on the specified agent if it belongs to that user.

        Args:
            user_id: The user whose default to change.
            agent_id: The agent to mark as default.
        """
        # Clear all defaults for this user
        await self._session.execute(
            update(AgentConfig)
            .where(AgentConfig.user_id == user_id)
            .values(is_default=False)
        )

        # Set the target agent as default (only if it belongs to this user)
        target = await self._session.execute(
            select(AgentConfig).where(
                AgentConfig.id == agent_id,
                AgentConfig.user_id == user_id,
            )
        )
        obj = target.scalar_one_or_none()
        if obj is not None:
            obj.is_default = True

        await self._session.commit()
