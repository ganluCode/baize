"""Repository for AgentConfig database operations."""

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from baize.agent.models import AgentConfig
from baize.agent.schemas import AgentCreate, AgentUpdate
from baize.user.models import UserModel


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
            agent_type=data.agent_type,
            prompts=data.prompts.model_dump(exclude_none=True) if data.prompts else None,
            model_config_json=data.llm_config,
            tools=data.tools.model_dump() if data.tools else None,
            sub_agents=data.sub_agents,
            memory_config=data.memory_config.model_dump() if data.memory_config else None,
            guardrails=data.guardrails.model_dump() if data.guardrails else None,
        )
        self._session.add(obj)
        await self._session.commit()
        await self._session.refresh(obj)
        return obj

    async def get_by_id(self, agent_id: uuid.UUID) -> AgentConfig | None:
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

    async def update(self, agent_id: uuid.UUID, data: AgentUpdate) -> AgentConfig | None:
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
        # llm_config maps to the ORM's model_config_json field
        if "llm_config" in update_data:
            update_data["model_config_json"] = update_data.pop("llm_config")

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

    async def get_default_by_user(self, user_id: uuid.UUID) -> AgentConfig | None:
        """Return the default agent config for a user, or None.

        Reads ``system_users.default_agent_id`` and loads the referenced agent.
        """
        user_result = await self._session.execute(
            select(UserModel.default_agent_id).where(UserModel.id == user_id)
        )
        default_id = user_result.scalar_one_or_none()
        if default_id is None:
            return None
        return await self.get_by_id(default_id)

    async def set_default(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> None:
        """Set a specific agent as the default for a user.

        Only sets the default if the agent belongs to the user.
        """
        # Verify the agent belongs to the user
        check = await self._session.execute(
            select(AgentConfig.id).where(
                AgentConfig.id == agent_id,
                AgentConfig.user_id == user_id,
            )
        )
        if check.scalar_one_or_none() is None:
            return

        await self._session.execute(
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(default_agent_id=agent_id)
        )
        await self._session.commit()

    async def clear_default_if_matches(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> None:
        """Clear the user's default_agent_id if it currently points to agent_id."""
        await self._session.execute(
            update(UserModel)
            .where(
                UserModel.id == user_id,
                UserModel.default_agent_id == agent_id,
            )
            .values(default_agent_id=None)
        )
        await self._session.commit()
