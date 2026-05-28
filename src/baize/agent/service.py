"""AgentConfigService and AgentService: business logic for agent management and chat."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from baize.agent.models import AgentConfig
from baize.agent.repository import AgentConfigRepository
from baize.agent.schemas import AgentCreate, AgentUpdate
from baize.agent.tools import ToolRegistry
from baize.session.service import SessionService

if TYPE_CHECKING:
    from baize.context.service import ContextService
    from baize.llm.model_router import ModelRouter
    from baize.memory.interface import MemoryServiceInterface
    from baize.user.models import UserModel

_PROMPTS_DIR = Path(__file__).parents[3] / "config" / "prompts"
_DEFAULT_SOUL_PATH = _PROMPTS_DIR / "default_soul.md"
_DEFAULT_BEHAVIOR_PATH = _PROMPTS_DIR / "default_behavior.md"


def _read_prompt_file(path: Path) -> str | None:
    """Read a prompt file, returning None if missing (with a warning)."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Prompt file not found at %s.", path)
        return None

_DEFAULT_AGENT_TOOLS = {
    "builtin": [
        "save_memory",
        "search_memory",
        "create_task",
        "list_tasks",
        "complete_task",
    ],
    "mcp_servers": [],
    "skills": [],
}

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


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

    def _validate_tools(self, tools) -> None:
        """Raise AgentServiceError if any builtin tool name is not registered.

        Args:
            tools: ToolsConfig or dict with builtin/mcp_servers/skills keys.

        Raises:
            AgentServiceError: 400 if any builtin name is not in the ToolRegistry.
        """
        if tools is None:
            return
        builtin = []
        if hasattr(tools, "builtin"):
            builtin = tools.builtin
        elif isinstance(tools, dict):
            builtin = tools.get("builtin", [])
        registered = set(ToolRegistry.get_all_names())
        invalid = [t for t in builtin if t not in registered]
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

        If ``data.set_as_default`` is True, the new agent becomes the user's default
        (overwriting ``system_users.default_agent_id``).

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

        if data.set_as_default:
            await self._repo.set_default(user_id, agent.id)

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

        If ``data.set_as_default`` is True, this agent becomes the user's default.

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
        await self.get(agent_id, user_id)

        if data.tools is not None:
            self._validate_tools(data.tools)

        updated = await self._repo.update(agent_id, data)
        assert updated is not None  # we verified it exists above

        if data.set_as_default is True:
            await self._repo.set_default(user_id, agent_id)

        logger.debug("Updated agent %s.", agent_id)
        return updated

    async def create_default_agent(self, user_id: uuid.UUID) -> AgentConfig:
        """Create the default 白泽（Baize）agent for a newly created user.

        Reads the layered prompts from ``config/prompts/default_soul.md`` and
        ``config/prompts/default_behavior.md``, assigns the standard tool set,
        and sets the new agent as default.

        Args:
            user_id: The user who will own the default agent.

        Returns:
            The newly created default AgentConfig.
        """
        from baize.agent.schemas import AgentPrompts, ToolsConfig

        prompts = AgentPrompts(
            soul=_read_prompt_file(_DEFAULT_SOUL_PATH),
            behavior=_read_prompt_file(_DEFAULT_BEHAVIOR_PATH),
        )

        data = AgentCreate(
            name="白泽（Baize）",
            description="你的个人 AI 助理",
            prompts=prompts,
            tools=ToolsConfig(**_DEFAULT_AGENT_TOOLS),
            set_as_default=True,
        )
        return await self.create(user_id, data)

    async def get_default(self, user_id: uuid.UUID) -> AgentConfig | None:
        """Return the default agent config for a user, or None if not found.

        Args:
            user_id: The user whose default agent to look up.

        Returns:
            The default AgentConfig, or None.
        """
        return await self._repo.get_default_by_user(user_id)

    async def delete(self, agent_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Delete an agent configuration and cascade-delete its sessions.

        If this agent is the user's default, ``default_agent_id`` is cleared.

        Args:
            agent_id: The agent to delete.
            user_id: The requesting user's id.

        Raises:
            AgentServiceError: 404 if not found or not owned by user.
        """
        await self.get(agent_id, user_id)

        # Clear default pointer if it referenced this agent
        await self._repo.clear_default_if_matches(user_id, agent_id)

        await self._session_service.delete_sessions_by_agent(agent_id)
        await self._repo.delete(agent_id)
        logger.debug("Deleted agent %s and its sessions.", agent_id)


# ---------------------------------------------------------------------------
# AgentService — chat orchestration
# ---------------------------------------------------------------------------


@dataclass
class ChatEvent:
    """A single streaming event from AgentService.chat().

    Attributes:
        type: Event kind — "token" | "tool_call" | "tool_result" | "done" | "error".
        payload: Event-specific data dict.
    """

    type: str
    payload: dict = field(default_factory=dict)


_MAX_TOOL_CALLS: int = 10
_CHAT_TIMEOUT: float = 120.0


class AgentService:
    """Orchestrates a complete chat turn: LLM resolution, LangGraph execution, and persistence.

    Context preparation (memory recall, system prompt assembly, history
    compression) is delegated to :class:`~baize.context.service.ContextService`.

    Args:
        agent_config_service: Used to load and validate AgentConfig.
        session_service: Used to load history, create sessions, and save messages.
        memory_service: Optional memory backend (passed through to graph state).
        model_router: Resolves the LLM instance to use.
        context_service: Prepares the layered context for each chat turn.
    """

    def __init__(
        self,
        agent_config_service: AgentConfigService,
        session_service: SessionService,
        memory_service: MemoryServiceInterface | None,
        model_router: ModelRouter,
        context_service: ContextService,
    ) -> None:
        self._agent_config_svc = agent_config_service
        self._session_svc = session_service
        self._memory_svc = memory_service
        self._model_router = model_router
        self._context_svc = context_service

    def _resolve_llm(self, agent_config: AgentConfig):
        """Resolve the chat LLM using the agent's model_config, falling back to 'default'."""
        ref: str | None = None
        if agent_config.model_config_json and isinstance(agent_config.model_config_json, dict):
            ref = agent_config.model_config_json.get("chat")
        if ref:
            provider_name, model_id = ref.split("/", 1)
            return self._model_router._factory.get_chat_model(provider_name, model_id)
        return self._model_router.get_chat_model("default")

    async def _get_or_create_session(self, session_id: uuid.UUID, user_id: uuid.UUID, agent_id: uuid.UUID):
        """Return the session if found, or create a new one if not found.

        Args:
            session_id: The requested session id.
            user_id: Owner user id.
            agent_id: Owner agent id.

        Returns:
            An existing or newly created SessionModel.
        """
        try:
            return await self._session_svc.get_session(session_id, user_id)
        except HTTPException as exc:
            if exc.status_code == 404:
                logger.debug("Session %s not found; creating a new one.", session_id)
                return await self._session_svc.create_session(user_id, agent_id)
            raise

    async def chat(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        message: str,
        user: UserModel,
        thinking: bool | None = None,
        external_history: list | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Execute a single chat turn and stream events to the caller.

        Orchestrates: agent loading → session setup → memory recall → prompt assembly →
        context compression → LangGraph execution → message persistence.

        Args:
            agent_id: The agent to chat with.
            session_id: Session to continue (auto-created if not found).
            user_id: The authenticated user's id.
            message: The user's input message.
            user: The authenticated UserModel (for prompt assembly).

        Yields:
            :class:`ChatEvent` instances of type "token", "tool_call", "tool_result",
            "done", or "error".
        """
        from baize.agent.pipeline import ContextStep, PersistStep, ReactStep, StepContext
        from baize.core.observability import TraceCollector

        _slog.info(
            "chat_request",
            user_id=str(user_id),
            session_id=str(session_id),
            agent_id=str(agent_id),
        )

        # 0. Create trace collector (spans reported in real time by each step)
        collector = TraceCollector(
            user_id=str(user_id),
            session_id=str(session_id),
            agent_id=str(agent_id),
        )

        # 1. Load agent config (validates ownership)
        try:
            agent_config = await self._agent_config_svc.get(agent_id, user_id)
        except AgentServiceError as exc:
            _slog.error("agent_error", error_type=type(exc).__name__,
                        error_message=str(exc), user_id=str(user_id),
                        session_id=str(session_id), agent_id=str(agent_id))
            collector.finalize(error=str(exc))
            yield ChatEvent(type="error", payload={"message": str(exc)})
            return

        collector.agent_name = agent_config.name

        # 2. Load or auto-create session
        session = await self._get_or_create_session(session_id, user_id, agent_id)
        collector.session_id = str(session.id)

        # 3. Resolve LLM
        llm = self._resolve_llm(agent_config)

        # 4. Build shared step context
        ctx = StepContext(
            collector=collector,
            user_id=str(user_id),
            session_id=str(session.id),
            agent_id=str(agent_id),
            message=message,
        )
        collector.set_input(message)

        # 5. Step: Context preparation (memory + prompt + history)
        context_step = ContextStep(
            context_svc=self._context_svc,
            agent_config=agent_config,
            user=user,
            llm=llm,
            external_history=external_history,
        )
        try:
            await context_step.run(ctx)
        except Exception as exc:
            collector.finalize(error=str(exc))
            yield ChatEvent(type="error", payload={"message": f"上下文准备失败: {exc}"})
            return

        # 6. Step: Graph execution (streams ChatEvents, reports LLM/tool spans)
        react_step = ReactStep(llm=llm, agent_config=agent_config, thinking=thinking)
        try:
            async for chat_event in react_step.stream(ctx):
                yield chat_event
                if chat_event.type == "error":
                    return
        except Exception as exc:
            collector.finalize(output=ctx.results.get("response", ""), error=str(exc))
            _slog.error("agent_error", error_type=type(exc).__name__,
                        error_message=str(exc), user_id=str(user_id),
                        session_id=str(session_id), agent_id=str(agent_id))
            raise

        # 7. Step: Persist messages
        persist_step = PersistStep(
            session_svc=self._session_svc,
            session_id=session.id,
            user_id=user_id,
        )
        await persist_step.run(ctx)

        yield ChatEvent(
            type="done",
            payload={
                "session_id": str(session.id),
                "message_id": ctx.results.get("message_id", ""),
            },
        )

        # 8. Finalize trace
        collector.finalize(output=ctx.results.get("response", ""))
