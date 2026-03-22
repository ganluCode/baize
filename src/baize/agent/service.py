"""AgentConfigService and AgentService: business logic for agent management and chat."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import structlog
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from baize.agent.context import compress_history
from baize.agent.graph import build_react_graph
from baize.agent.models import AgentConfig
from baize.agent.prompt import assemble_system_prompt
from baize.agent.repository import AgentConfigRepository
from baize.agent.schemas import AgentCreate, AgentResponse, AgentUpdate
from baize.agent.tools import ToolRegistry
from baize.session.models import MessageRole
from baize.session.service import SessionService

if TYPE_CHECKING:
    from baize.llm.model_router import ModelRouter
    from baize.memory.interface import Memory, MemoryServiceInterface
    from baize.user.models import UserModel

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


def _history_to_lc_messages(messages):
    """Convert a list of ChatMessageModel to LangChain BaseMessage objects."""
    from baize.session.models import ChatMessageModel

    result = []
    for msg in messages:
        if msg.role == MessageRole.user:
            result.append(HumanMessage(content=msg.content))
        elif msg.role == MessageRole.assistant:
            result.append(AIMessage(content=msg.content))
        elif msg.role == MessageRole.system:
            result.append(SystemMessage(content=msg.content))
        # tool-role messages are skipped (not directly representable for replay)
    return result


class AgentService:
    """Orchestrates a complete chat turn: prompt assembly, LangGraph execution, and persistence.

    Args:
        agent_config_service: Used to load and validate AgentConfig.
        session_service: Used to load history, create sessions, and save messages.
        memory_service: Optional memory backend for auto-recall.
        model_router: Resolves the LLM instance to use.
    """

    def __init__(
        self,
        agent_config_service: AgentConfigService,
        session_service: SessionService,
        memory_service: "MemoryServiceInterface | None",
        model_router: "ModelRouter",
    ) -> None:
        self._agent_config_svc = agent_config_service
        self._session_svc = session_service
        self._memory_svc = memory_service
        self._model_router = model_router

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
        user: "UserModel",
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
        _slog.info(
            "chat_request",
            user_id=str(user_id),
            session_id=str(session_id),
            agent_id=str(agent_id),
        )

        # 1. Load agent config (validates ownership)
        try:
            agent_config = await self._agent_config_svc.get(agent_id, user_id)
        except AgentServiceError as exc:
            _slog.error(
                "agent_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                user_id=str(user_id),
                session_id=str(session_id),
                agent_id=str(agent_id),
            )
            yield ChatEvent(type="error", payload={"message": str(exc)})
            return

        # 2. Load or auto-create session
        session = await self._get_or_create_session(session_id, user_id, agent_id)

        # 2b. Resolve memory configuration (used both for auto-recall and tool injection)
        from baize.core.config import get_settings
        from baize.memory.service import resolve_memory_config

        resolved_memory = resolve_memory_config(
            session=session,
            agent_config=agent_config,
            user=user,
            global_config=get_settings(),
        )

        # 3. Memory recall (if enabled)
        memories: list[Memory] = []
        if resolved_memory.auto_memory_recall and self._memory_svc is not None:
            try:
                memories = await self._memory_svc.search(message, top_k=5)
            except Exception:
                logger.warning("Memory recall failed; continuing without memories.", exc_info=True)

        # 4. Assemble system prompt
        system_prompt = assemble_system_prompt(agent_config, user, memories)

        # 5. Load and compress conversation history
        history_models = await self._session_svc.get_history(session.id)
        lc_history = _history_to_lc_messages(history_models)

        llm = self._model_router.get_chat_model("default")
        compressed_history = await compress_history(lc_history, llm)

        # 6. Resolve tools
        tool_names: list[str] = list(agent_config.tools or [])
        tool_entries = ToolRegistry.get_by_names(tool_names)
        tools = [e.langchain_tool for e in tool_entries if e.langchain_tool is not None]

        # 7. Build ReAct graph (fresh per request)
        graph = build_react_graph(llm, tools)

        # 8. Compose initial messages
        initial_messages = [
            SystemMessage(content=system_prompt),
            *compressed_history,
            HumanMessage(content=message),
        ]

        # Build LangGraph run config (inject LangFuse handler when available)
        from baize.core.observability import create_langfuse_handler

        langfuse_handler = create_langfuse_handler()
        run_config: dict = {
            "configurable": {"thread_id": str(session.id)},
            "metadata": {
                "user_id": str(user_id),
                "session_id": str(session_id),
                "agent_id": str(agent_id),
            },
        }
        if langfuse_handler is not None:
            run_config["callbacks"] = [langfuse_handler]

        tool_call_count = 0
        current_response = ""
        _llm_start_time: float = 0.0
        _tool_start_times: dict[str, float] = {}

        # 9. Stream with timeout guard
        try:
            async with asyncio.timeout(_CHAT_TIMEOUT):
                async for event in graph.astream_events(
                    {
                        "messages": initial_messages,
                        "user_id": str(user_id),
                        "agent_id": str(agent_id),
                        "shared_memory": resolved_memory.shared_memory,
                    },
                    config=run_config,
                    version="v2",
                ):
                    event_type = event.get("event", "")

                    if event_type == "on_chat_model_start":
                        current_response = ""
                        _llm_start_time = time.monotonic()

                    elif event_type == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk is not None and hasattr(chunk, "content"):
                            content = chunk.content
                            if isinstance(content, str) and content:
                                current_response += content
                                yield ChatEvent(type="token", payload={"content": content})

                    elif event_type == "on_chat_model_end":
                        latency_ms = int((time.monotonic() - _llm_start_time) * 1000)
                        output = event.get("data", {}).get("output")
                        usage = getattr(output, "usage_metadata", None) or {}
                        _slog.debug(
                            "llm_call",
                            model=event.get("name", ""),
                            prompt_tokens=usage.get("input_tokens", 0),
                            completion_tokens=usage.get("output_tokens", 0),
                            latency_ms=latency_ms,
                        )

                    elif event_type == "on_tool_start":
                        tool_call_count += 1
                        tool_name = event.get("name", "")
                        _tool_start_times[tool_name] = time.monotonic()
                        if tool_call_count > _MAX_TOOL_CALLS:
                            yield ChatEvent(
                                type="error",
                                payload={"message": f"工具调用次数超过上限（{_MAX_TOOL_CALLS}次）。"},
                            )
                            return
                        yield ChatEvent(
                            type="tool_call",
                            payload={
                                "tool": tool_name,
                                "args": event["data"].get("input", {}),
                            },
                        )

                    elif event_type == "on_tool_end":
                        tool_name = event.get("name", "")
                        start = _tool_start_times.pop(tool_name, time.monotonic())
                        duration_ms = int((time.monotonic() - start) * 1000)
                        output = event["data"].get("output", "")
                        output_str = output.content if hasattr(output, "content") else str(output)
                        _slog.info("tool_call", tool_name=tool_name, duration_ms=duration_ms)
                        yield ChatEvent(
                            type="tool_result",
                            payload={"tool": tool_name, "result": output_str},
                        )

        except TimeoutError:
            yield ChatEvent(type="error", payload={"message": "对话超时（120秒），请稍后重试。"})
            return
        except Exception as exc:
            _slog.error(
                "agent_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                user_id=str(user_id),
                session_id=str(session_id),
                agent_id=str(agent_id),
            )
            raise

        # 10. Persist messages
        user_msg = await self._session_svc.save_message(
            session_id=session.id,
            user_id=user_id,
            role=MessageRole.user,
            content=message,
        )
        await self._session_svc.save_message(
            session_id=session.id,
            user_id=user_id,
            role=MessageRole.assistant,
            content=current_response,
        )

        yield ChatEvent(
            type="done",
            payload={"session_id": str(session.id), "message_id": str(user_msg.id)},
        )
