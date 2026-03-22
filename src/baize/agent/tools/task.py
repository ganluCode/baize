"""Built-in task/checklist agent tools: create_task, list_tasks, complete_task.

All three tools are registered with permission='auto' and delegate all work to
the TaskService obtained via :func:`_get_task_service`.  When the task service
is not yet initialised the tools return a user-friendly message instead of
raising an exception.
"""

from __future__ import annotations

import logging
from typing import Any

from baize.agent.tools import register_tool

logger = logging.getLogger(__name__)


def _get_task_service() -> Any | None:
    """Return the TaskService from the global container, or None.

    Avoids hard-importing the container at module load time; the container may
    not be initialised when this module is first imported.
    """
    try:
        from baize.core.deps import get_container

        container = get_container()
        return container.task_service
    except RuntimeError:
        # Container not yet initialised (e.g. during testing without lifespan).
        return None


@register_tool(permission="auto", description="Create a new task in the checklist.")
async def create_task(
    title: str,
    due_date: str | None = None,
    priority: str = "medium",
) -> str:
    """Create a new task in the checklist.

    Args:
        title: Title of the task.
        due_date: Optional due date string (e.g. "2026-03-25").
        priority: Priority level — "low", "medium", or "high" (default "medium").

    Returns:
        Confirmation string with the new task ID, or a status message.
    """
    svc = _get_task_service()
    if svc is None:
        return "任务服务尚未初始化，无法创建任务。"

    task = await svc.create(title=title, due_date=due_date, priority=priority)
    return f"已创建任务（ID: {task.id}）：{title}"


@register_tool(permission="auto", description="List tasks from the checklist.")
async def list_tasks(status: str = "pending") -> str:
    """List tasks filtered by status.

    Args:
        status: Task status to filter by — "pending", "completed", etc.
                Defaults to "pending".

    Returns:
        Formatted string listing matching tasks, or a status message.
    """
    svc = _get_task_service()
    if svc is None:
        return "任务服务尚未初始化，无法查询任务。"

    tasks = await svc.list_by_user(status=status)

    if not tasks:
        return f"没有{status}状态的任务。"

    lines = [f"## 任务列表（{status}）"]
    for i, task in enumerate(tasks, start=1):
        priority_label = getattr(task, "priority", "medium")
        due = getattr(task, "due_date", None)
        due_str = f"，截止：{due}" if due else ""
        lines.append(f"{i}. [{task.id}] {task.title}（优先级：{priority_label}{due_str}）")

    return "\n".join(lines)


@register_tool(permission="auto", description="Mark a task as completed.")
async def complete_task(task_id: str) -> str:
    """Mark a task as completed.

    Args:
        task_id: The ID of the task to complete.

    Returns:
        Confirmation string, or an error description if the task does not exist.
    """
    svc = _get_task_service()
    if svc is None:
        return "任务服务尚未初始化，无法完成任务。"

    try:
        await svc.complete(task_id)
        return f"任务 {task_id} 已标记为完成。"
    except Exception as exc:  # noqa: BLE001
        logger.warning("complete_task failed for task_id=%s: %s", task_id, exc)
        return f"完成任务失败：{exc}"
