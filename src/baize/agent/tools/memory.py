"""Built-in memory agent tools: save_memory and search_memory.

Both tools are registered with permission='auto' and delegate all work to the
MemoryServiceInterface obtained via :func:`_get_memory_service`.  When the
memory service is not yet initialised the tools return a user-friendly message
instead of raising an exception.
"""

from __future__ import annotations

import logging
from typing import Any

from baize.agent.tools import register_tool

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


@register_tool(permission="auto", description="Save a piece of information to long-term memory.")
async def save_memory(content: str, metadata: dict[str, Any] | None = None) -> str:
    """Save a piece of information to long-term memory.

    Args:
        content: The text content to remember.
        metadata: Optional key-value metadata to attach to the memory.

    Returns:
        Confirmation string with the stored memory ID, or an error description.
    """
    svc = _get_memory_service()
    if svc is None:
        return "记忆服务尚未初始化，无法保存记忆。"

    if metadata is None:
        metadata = {}

    try:
        memory_id = await svc.add_memory(content, metadata)
        return f"已保存记忆（ID: {memory_id}）：{content}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_memory failed: %s", exc)
        return f"保存记忆失败：{exc}"


@register_tool(permission="auto", description="Search long-term memory for relevant information.")
async def search_memory(query: str, top_k: int = 5) -> str:
    """Search long-term memory for information relevant to a query.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of memories to return (default 5).

    Returns:
        Formatted string listing matching memories, or a status message.
    """
    svc = _get_memory_service()
    if svc is None:
        return "记忆服务尚未初始化，无法搜索记忆。"

    memories = await svc.search(query, top_k=top_k)

    if not memories:
        return "未找到相关记忆。"

    lines = ["## 相关记忆"]
    for i, mem in enumerate(memories, start=1):
        lines.append(f"{i}. {mem.content}")

    return "\n".join(lines)
