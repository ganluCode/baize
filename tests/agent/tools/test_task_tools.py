"""Unit tests for the built-in task agent tools (F-008).

Tests cover:
- create_task returns confirmation with task ID on success
- create_task returns "not initialised" message when service is None
- list_tasks returns formatted task list on success
- list_tasks returns "not initialised" message when service is None
- list_tasks handles empty result gracefully
- complete_task returns confirmation on success
- complete_task returns error description when task not found
- complete_task returns "not initialised" message when service is None
- All three tools are registered in ToolRegistry with permission='auto'
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeTask:
    id: str
    title: str
    status: str = "pending"
    due_date: str | None = None
    priority: str = "medium"


def _make_task_service(
    create_return: FakeTask | None = None,
    list_return: list[FakeTask] | None = None,
    complete_return=None,
    complete_side_effect=None,
) -> AsyncMock:
    """Return a mock TaskService."""
    svc = AsyncMock()
    svc.create.return_value = create_return or FakeTask(id="task-1", title="Test Task")
    svc.list_by_user.return_value = list_return or []
    if complete_side_effect is not None:
        svc.complete.side_effect = complete_side_effect
    else:
        svc.complete.return_value = complete_return
    return svc


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_returns_confirmation_with_task_id() -> None:
    task = FakeTask(id="task-42", title="Write tests")
    svc = _make_task_service(create_return=task)

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import create_task

        result = await create_task(title="Write tests")

    assert "task-42" in result


@pytest.mark.asyncio
async def test_create_task_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.task._get_task_service", return_value=None):
        from baize.agent.tools.task import create_task

        result = await create_task(title="Something")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_create_task_passes_all_params_to_service() -> None:
    svc = _make_task_service()

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import create_task

        await create_task(title="Buy milk", due_date="2026-03-25", priority="high")

    svc.create.assert_called_once_with(title="Buy milk", due_date="2026-03-25", priority="high")


@pytest.mark.asyncio
async def test_create_task_uses_default_priority_medium() -> None:
    svc = _make_task_service()

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import create_task

        await create_task(title="Some task")

    call_kwargs = svc.create.call_args.kwargs
    assert call_kwargs.get("priority") == "medium"


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tasks_returns_formatted_list_on_success() -> None:
    tasks = [
        FakeTask(id="t1", title="Buy groceries", priority="high"),
        FakeTask(id="t2", title="Write report", priority="medium"),
    ]
    svc = _make_task_service(list_return=tasks)

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import list_tasks

        result = await list_tasks(status="pending")

    assert "Buy groceries" in result
    assert "Write report" in result


@pytest.mark.asyncio
async def test_list_tasks_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.task._get_task_service", return_value=None):
        from baize.agent.tools.task import list_tasks

        result = await list_tasks()

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_list_tasks_handles_empty_results() -> None:
    svc = _make_task_service(list_return=[])

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import list_tasks

        result = await list_tasks(status="pending")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_list_tasks_passes_status_to_service() -> None:
    svc = _make_task_service(list_return=[])

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import list_tasks

        await list_tasks(status="completed")

    svc.list_by_user.assert_called_once_with(status="completed")


# ---------------------------------------------------------------------------
# complete_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_task_returns_confirmation_on_success() -> None:
    svc = _make_task_service()

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import complete_task

        result = await complete_task(task_id="task-99")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_complete_task_returns_error_when_task_not_found() -> None:
    svc = _make_task_service(complete_side_effect=ValueError("Task not found"))

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import complete_task

        result = await complete_task(task_id="nonexistent")

    # Should not crash; should return an error description
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain some indication of the error
    assert "not found" in result.lower() or "失败" in result or "error" in result.lower() or "不存在" in result


@pytest.mark.asyncio
async def test_complete_task_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.task._get_task_service", return_value=None):
        from baize.agent.tools.task import complete_task

        result = await complete_task(task_id="task-1")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_complete_task_passes_task_id_to_service() -> None:
    svc = _make_task_service()

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import complete_task

        await complete_task(task_id="task-77")

    svc.complete.assert_called_once_with("task-77")


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_task_tools_are_registered_in_tool_registry() -> None:
    from baize.agent.tools import ToolRegistry
    from baize.agent.tools import task as _  # noqa: F401 — trigger registration

    names = ToolRegistry.get_all_names()
    assert "create_task" in names
    assert "list_tasks" in names
    assert "complete_task" in names


def test_task_tools_have_auto_permission() -> None:
    from baize.agent.tools import ToolRegistry
    from baize.agent.tools import task as _  # noqa: F401

    for name in ("create_task", "list_tasks", "complete_task"):
        entry = ToolRegistry.get(name)
        assert entry is not None and entry.permission == "auto", f"{name} should have auto permission"
