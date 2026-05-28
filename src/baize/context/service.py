"""ContextService — orchestrates context preparation for one chat turn.

Responsibilities:
    1. Resolve memory strategy (agent > global).
    2. Perform memory recall when enabled.
    3. Assemble layered system prompt.
    4. Load conversation history and compress it.

The LLM is supplied by the caller (``AgentService``) because routing belongs
to agent orchestration, not context engineering.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from baize.context.history import compress_history
from baize.context.memory_config import resolve_memory_config
from baize.context.schemas import PreparedContext
from baize.context.system_prompt import assemble_system_prompt
from baize.session.models import MessageRole

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel

    from baize.agent.models import AgentConfig
    from baize.core.config import Settings
    from baize.memory.interface import Memory, MemoryServiceInterface
    from baize.session.service import SessionService
    from baize.user.models import UserModel

logger = logging.getLogger(__name__)


def _history_to_lc_messages(messages) -> list[BaseMessage]:
    """Convert a list of ChatMessageModel to LangChain BaseMessage objects."""
    result: list[BaseMessage] = []
    for msg in messages:
        if msg.role == MessageRole.user:
            result.append(HumanMessage(content=msg.content))
        elif msg.role == MessageRole.assistant:
            result.append(AIMessage(content=msg.content))
        elif msg.role == MessageRole.system:
            result.append(SystemMessage(content=msg.content))
        # tool-role messages are skipped (not directly representable for replay)
    return result


class ContextService:
    """Prepare all context inputs for one chat turn.

    Resolves memory strategy, performs memory recall, assembles the layered
    system prompt, loads conversation history, and compresses it.
    """

    def __init__(
        self,
        session_service: SessionService,
        memory_service: MemoryServiceInterface | None,
        settings: Settings,
    ) -> None:
        self._session_svc = session_service
        self._memory_svc = memory_service
        self._settings = settings

    async def prepare(
        self,
        *,
        agent_config: AgentConfig,
        user: UserModel,
        session_id: uuid.UUID,
        message: str,
        llm: BaseLanguageModel,
        external_history: list[BaseMessage] | None = None,
    ) -> PreparedContext:
        """Prepare the full context for one chat turn.

        Args:
            agent_config: The agent whose prompt/tools/memory settings apply.
            user: The current user (used for preference layer).
            session_id: Session whose history should be loaded (ignored if
                ``external_history`` is provided).
            message: The current user message — used for memory recall query.
            llm: LangChain chat model used by history compression.
            external_history: When provided, use this list as the conversation
                history instead of loading from DB. Used by stateless clients
                (CherryStudio / OpenClaw / Open WebUI) that send the full
                ``messages`` array on every request.

        Returns:
            :class:`PreparedContext` bundling system prompt, compressed history,
            recalled memories and resolved memory configuration.
        """
        # 1. Resolve memory configuration
        resolved_memory = resolve_memory_config(agent_config, self._settings)

        # 2. Memory recall (if enabled)
        memories: list[Memory] = []
        if resolved_memory.auto_memory_recall and self._memory_svc is not None:
            try:
                memories = await self._memory_svc.search(
                    message, user_id=str(user.id), top_k=resolved_memory.top_k
                )
            except Exception:
                logger.warning(
                    "Memory recall failed; continuing without memories.",
                    exc_info=True,
                )

        # 3. Assemble layered system prompt
        system_prompt = assemble_system_prompt(agent_config, user, memories)

        # 4. Load and compress conversation history
        if external_history is not None:
            # Stateless mode — use client-provided history, skip DB lookup
            lc_history = external_history
        else:
            history_models = await self._session_svc.get_history(session_id)
            lc_history = _history_to_lc_messages(history_models)
        compressed_history = await compress_history(lc_history, llm)

        return PreparedContext(
            system_prompt=system_prompt,
            history=compressed_history,
            resolved_memory=resolved_memory,
            memories=memories,
        )
