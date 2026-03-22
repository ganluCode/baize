"""Built-in task/checklist agent tools: create_task, list_tasks, complete_task.

All three tools are registered with permission='auto' and delegate all work to
the TaskEngineService obtained via :func:`_get_task_service`.  When the task
service is not yet initialised the tools return a user-friendly message instead
of raising an exception.

user_id is injected by the LangGraph framework (InjectedState) and is NOT
exposed to the LLM in the tool schema.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Annotated, Any

from langgraph.prebuilt import InjectedState

from baize.agent.tools import register_tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _get_task_service() -> Any | None:
    """Return the TaskEngineService from the global container, or None.

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


@register_tool(permission="auto", description="在任务清单中为当前用户创建一个新任务")
async def create_task(
    title: str,
    state: Annotated[dict, InjectedState],
    due_date: str | None = None,
    priority: str = "medium",
) -> str:
    """在任务清单中为当前用户创建一个新任务。

    Args:
        title: 任务标题（必填）。
        state: 注入的图状态，包含 user_id。不暴露给 LLM。
        due_date: 可选截止日期字符串，格式为 "YYYY-MM-DD"。
        priority: 优先级，可选 "low"、"medium"（默认）或 "high"。

    Returns:
        包含任务 ID 和标题的确认字符串，或错误描述。
    """
    svc = _get_task_service()
    if svc is None:
        return "任务服务尚未初始化，无法创建任务。"

    user_id_str: str = state.get("user_id", "")
    if not user_id_str:
        return "无法获取用户信息，无法创建任务。"

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return "用户 ID 格式无效，无法创建任务。"

    try:
        from baize.task.models import TaskSource

        task = await svc.create_task(
            user_id=user_id,
            title=title,
            due_date=due_date,
            priority=priority,
            source=TaskSource.agent,
        )
        due_str = f"，截止：{task.due_date}" if task.due_date else ""
        return f"已创建任务（ID: {task.id}）：{title}{due_str}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_task failed: %s", exc)
        return f"创建任务失败：{exc}"


@register_tool(permission="auto", description="查询当前用户的任务清单，可按状态过滤")
async def list_tasks(
    state: Annotated[dict, InjectedState],
    status: str = "todo",
) -> str:
    """查询当前用户的任务清单，可按状态过滤。

    Args:
        state: 注入的图状态，包含 user_id。不暴露给 LLM。
        status: 按状态过滤，可选 "todo"（默认）、"in_progress"、"done" 或 "all"（不过滤）。

    Returns:
        格式化的任务列表字符串，或提示信息。
    """
    svc = _get_task_service()
    if svc is None:
        return "任务服务尚未初始化，无法查询任务。"

    user_id_str: str = state.get("user_id", "")
    if not user_id_str:
        return "无法获取用户信息，无法查询任务。"

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return "用户 ID 格式无效，无法查询任务。"

    tasks = await svc.list_tasks(user_id=user_id, status=status if status != "all" else None)

    if not tasks:
        return "暂无待办任务。" if status == "todo" else f"暂无{status}状态的任务。"

    status_label = "全部" if status == "all" else status
    lines = [f"## 任务列表（{status_label}）"]
    for task in tasks:
        due_str = str(task.due_date) if task.due_date else "无"
        lines.append(
            f"- {task.title} | 优先级: {task.priority.value} | 截止: {due_str} | 状态: {task.status.value}"
        )
    return "\n".join(lines)


@register_tool(permission="auto", description="通过标题模糊匹配，将当前用户的一个待办任务标记为完成")
async def complete_task(
    task_title: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """通过标题模糊匹配，将当前用户的一个待办任务标记为完成。

    Args:
        task_title: 要完成的任务标题（支持模糊匹配）。
        state: 注入的图状态，包含 user_id。不暴露给 LLM。

    Returns:
        完成确认字符串、候选任务列表或"未找到"提示。
    """
    svc = _get_task_service()
    if svc is None:
        return "任务服务尚未初始化，无法完成任务。"

    user_id_str: str = state.get("user_id", "")
    if not user_id_str:
        return "无法获取用户信息，无法完成任务。"

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return "用户 ID 格式无效，无法完成任务。"

    try:
        matches = await svc.search_todo_by_title(user_id=user_id, title=task_title)

        if not matches:
            return f'未找到包含 "{task_title}" 的待办任务。'

        if len(matches) > 1:
            candidates = "\n".join(f"- {t.title}（ID: {t.id}）" for t in matches)
            return f'找到多个匹配 "{task_title}" 的待办任务，请确认要完成哪一个：\n{candidates}'

        task = matches[0]
        await svc.mark_done(user_id=user_id, task_id=task.id)
        return f'任务 "{task.title}" 已标记为完成。'
    except Exception as exc:  # noqa: BLE001
        logger.warning("complete_task failed: %s", exc)
        return f"完成任务失败：{exc}"
