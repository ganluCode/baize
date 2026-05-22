"""Built-in memory agent tools: save_memory and search_memory.

Both tools are registered with permission='auto' and delegate all work to the
MemoryServiceInterface obtained via :func:`_get_memory_service`.  When the
memory service is not yet initialised the tools return a user-friendly message
instead of raising an exception.

user_id, agent_id, session_id and shared_memory are injected by the LangGraph
framework (InjectedState / RunnableConfig) and are NOT exposed to the LLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState

from baize.agent.tools import register_tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _get_memory_service() -> Any | None:
    """Return the MemoryService from the global container, or None.

    Avoids hard-importing the container at module load time; the container may
    not be initialised when this module is first imported.
    """
    try:
        from baize.core.deps import get_container

        container = get_container()
        return container.memory_service
    except RuntimeError:
        # Container not yet initialised (e.g. during testing without lifespan).
        return None


@register_tool(permission="auto", description="保存一条重要信息到长期记忆")
async def save_memory(
    content: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
) -> str:
    """保存一条重要信息到长期记忆。

    Args:
        content: The text content to remember.
        state: Injected graph state — provides user_id, agent_id, shared_memory.
               Not exposed to the Agent in the tool schema.
        config: Injected runnable config — provides session_id via thread_id.
                Not exposed to the Agent in the tool schema.

    Returns:
        Confirmation string with the stored memory ID, or an error description.
    """
    if not content or not content.strip():
        return "content 不能为空，无法保存记忆。"

    svc = _get_memory_service()
    if svc is None:
        return "记忆服务尚未初始化，无法保存记忆。"

    user_id: str = state.get("user_id", "")
    agent_id: str | None = state.get("agent_id")
    shared: bool = state.get("shared_memory", True)
    configurable: dict = (config or {}).get("configurable", {})  # type: ignore[union-attr]
    session_id: str | None = configurable.get("thread_id")

    try:
        memory_id = await svc.add(
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            shared=shared,
        )
        return f"已保存记忆（ID: {memory_id}）：{content}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_memory failed: %s", exc)
        return f"保存记忆失败：{exc}"


@register_tool(permission="auto", description="从长期记忆中搜索与问题相关的信息")
async def search_memory(
    query: str,
    state: Annotated[dict, InjectedState],
    top_k: int = 5,
) -> str:
    """从长期记忆中搜索与查询相关的信息。

    Args:
        query: Natural-language search query.
        state: Injected graph state — provides user_id and agent_id.
               Not exposed to the Agent in the tool schema.
        top_k: Maximum number of memories to return (default 5).

    Returns:
        Formatted string listing matching memories with scores, or a status message.
    """
    svc = _get_memory_service()
    if svc is None:
        return "记忆服务尚未初始化，无法搜索记忆。"

    user_id: str = state.get("user_id", "")
    agent_id: str | None = state.get("agent_id")

    memories = await svc.search(query, user_id=user_id, agent_id=agent_id, top_k=top_k)

    if not memories:
        return "未找到相关记忆。"

    lines = ["## 相关记忆"]
    for i, mem in enumerate(memories, start=1):
        score_str = f"（相关性：{mem.score:.2f}）" if mem.score is not None else ""
        lines.append(f"{i}. {mem.content}{score_str}")

    return "\n".join(lines)
