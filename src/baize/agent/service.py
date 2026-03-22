"""AgentConfigService: business logic for agent configuration management."""

import logging
import uuid
from pathlib import Path

from baize.agent.models import AgentConfig
from baize.agent.repository import AgentConfigRepository
from baize.agent.schemas import AgentCreate, AgentResponse, AgentUpdate
from baize.agent.tools import ToolRegistry
from baize.session.service import SessionService

_DEFAULT_AGENT_PROMPT_PATH = (
    Path(__file__).parents[3] / "config" / "prompts" / "default_agent.md"
)

_DEFAULT_AGENT_TOOLS = [
    "save_memory",
    "search_memory",
    "create_task",
    "list_tasks",
    "complete_task",
]

logger = logging.getLogger(__name__)


class AgentServiceError(Exception):
    """Raised when an agent service operation fails."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class AgentConfigService:
    """Business logic layer for AgentConfig CRUD with constraint enforcement."""

    def __init__(
        self,
        repository: AgentConfigRepository,
        session_service: SessionService,
    ) -> None:
        self._repo = repository
        self._session_service = session_service

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_tools(self, tools: list[str]) -> None:
        """Raise AgentServiceError if any tool name is not registered.

        Args:
            tools: List of tool names to validate.

        Raises:
            AgentServiceError: 400 if any name is not in the ToolRegistry.
        """
        registered = set(ToolRegistry.get_all_names())
        invalid = [t for t in tools if t not in registered]
        if invalid:
            raise AgentServiceError(
                f"Unknown tools: {', '.join(sorted(invalid))}. "
                f"Registered tools: {', '.join(sorted(registered)) or '(none)'}.",
                status_code=400,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create(self, user_id: uuid.UUID, data: AgentCreate) -> AgentConfig:
        """Create a new agent configuration.

        If ``data.is_default`` is True, the existing default agent for the user
        is demoted before the new one is created.

        Args:
            user_id: The user who owns the new agent.
            data: Validated creation payload.

        Returns:
            The newly created AgentConfig.

        Raises:
            AgentServiceError: 400 if any tool in ``data.tools`` is not registered.
        """
        self._validate_tools(data.tools)

        agent = await self._repo.create(user_id, data)
        logger.debug("Created agent %s for user %s.", agent.id, user_id)

        if data.is_default:
            await self._repo.set_default(user_id, agent.id)
            agent.is_default = True

        return agent

    async def get(self, agent_id: uuid.UUID, user_id: uuid.UUID) -> AgentConfig:
        """Return an agent config, enforcing ownership.

        Args:
            agent_id: The agent to retrieve.
            user_id: The requesting user's id.

        Returns:
            The matching AgentConfig.

        Raises:
            AgentServiceError: 404 if not found or not owned by the user.
        """
        agent = await self._repo.get_by_id(agent_id)
        if agent is None or agent.user_id != user_id:
            raise AgentServiceError("Agent not found.", status_code=404)
        return agent

    async def list(self, user_id: uuid.UUID) -> list[AgentConfig]:
        """Return all agent configs for a user.

        Args:
            user_id: The user whose agents to list.

        Returns:
            List of AgentConfig ordered by created_at ascending.
        """
        return await self._repo.list_by_user(user_id)

    async def update(
        self, agent_id: uuid.UUID, user_id: uuid.UUID, data: AgentUpdate
    ) -> AgentConfig:
        """Partially update an agent configuration.

        If ``data.is_default`` is True, this agent becomes the user's default
        (demoting the previous default). Setting ``is_default=False`` on the
        current default agent is a no-op — the default can only be changed by
        promoting another agent.

        Args:
            agent_id: The agent to update.
            user_id: The requesting user's id.
            data: Partial update payload.

        Returns:
            The updated AgentConfig.

        Raises:
            AgentServiceError: 404 if not found or not owned by user.
            AgentServiceError: 400 if any new tool name is not registered.
        """
        agent = await self.get(agent_id, user_id)

        if data.tools is not None:
            self._validate_tools(data.tools)

        updated = await self._repo.update(agent_id, data)
        assert updated is not None  # we verified it exists above

        if data.is_default is True:
            await self._repo.set_default(user_id, agent_id)
            updated.is_default = True

        logger.debug("Updated agent %s.", agent_id)
        return updated

    async def create_default_agent(self, user_id: uuid.UUID) -> AgentConfig:
        """Create the default 白泽（Baize）agent for a newly created user.

        Reads the system prompt from ``config/prompts/default_agent.md`` and
        creates an agent with the standard tool set and ``is_default=True``.

        Args:
            user_id: The user who will own the default agent.

        Returns:
            The newly created default AgentConfig.
        """
        try:
            system_prompt = _DEFAULT_AGENT_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError:
            logger.warning(
                "Default agent prompt not found at %s; using empty prompt.",
                _DEFAULT_AGENT_PROMPT_PATH,
            )
            system_prompt = ""

        data = AgentCreate(
            name="白泽（Baize）",
            description="你的个人 AI 助理",
            system_prompt=system_prompt,
            tools=_DEFAULT_AGENT_TOOLS,
            is_default=True,
        )
        return await self.create(user_id, data)

    async def delete(self, agent_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Delete an agent configuration and cascade-delete its sessions.

        The default agent cannot be deleted while it is still the only or the
        designated default — callers must promote another agent first.

        Args:
            agent_id: The agent to delete.
            user_id: The requesting user's id.

        Raises:
            AgentServiceError: 404 if not found or not owned by user.
            AgentServiceError: 400 if the agent is the user's default.
        """
        agent = await self.get(agent_id, user_id)

        if agent.is_default:
            raise AgentServiceError(
                "Cannot delete the default agent. Promote another agent as default first.",
                status_code=400,
            )

        await self._session_service.delete_sessions_by_agent(agent_id)
        await self._repo.delete(agent_id)
        logger.debug("Deleted agent %s and its sessions.", agent_id)
