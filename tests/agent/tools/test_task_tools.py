"""Unit tests for the built-in task agent tools (F-007, F-008, F-009).

Tests cover:
- create_task returns confirmation with task ID and title on success
- create_task returns "not initialised" message when service is None
- create_task sets source=agent via the service call
- list_tasks returns formatted task list on success
- list_tasks returns "not initialised" message when service is None
- list_tasks handles empty result gracefully
- list_tasks passes status='all' as None to service
- complete_task returns confirmation when exactly one task matches
- complete_task returns candidate list when multiple tasks match
- complete_task returns "not found" message when no task matches
- complete_task returns "not initialised" message when service is None
- All three tools are registered in ToolRegistry with permission='auto'
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeTask:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    title: str = "Test Task"
    status: str = "todo"
    due_date: date | None = None
    priority: object = None

    def __post_init__(self):
        from baize.task.models import TaskPriority, TaskStatus

        if self.priority is None:
            self.priority = TaskPriority.medium
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)


def _state(user_id: str = str(uuid.uuid4())) -> dict:
    """Build a minimal AgentState dict for tool injection."""
    return {"user_id": user_id}


def _make_task_service(
    create_return: FakeTask | None = None,
    list_return: list[FakeTask] | None = None,
    search_return: list[FakeTask] | None = None,
    mark_done_return: FakeTask | None = None,
    mark_done_side_effect=None,
) -> AsyncMock:
    """Return a mock TaskEngineService."""
    svc = AsyncMock()
    svc.create_task.return_value = create_return or FakeTask(title="Test Task")
    svc.list_tasks.return_value = list_return or []
    svc.search_todo_by_title.return_value = search_return or []
    if mark_done_side_effect is not None:
        svc.mark_done.side_effect = mark_done_side_effect
    else:
        svc.mark_done.return_value = mark_done_return or FakeTask(title="Test Task")
    return svc


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_returns_confirmation_with_task_id() -> None:
    task = FakeTask(id=uuid.uuid4(), title="Write tests")
    svc = _make_task_service(create_return=task)

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import create_task

        result = await create_task(title="Write tests", state=_state())

    assert str(task.id) in result
    assert "Write tests" in result


@pytest.mark.asyncio
async def test_create_task_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.task._get_task_service", return_value=None):
        from baize.agent.tools.task import create_task

        result = await create_task(title="Something", state=_state())

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_create_task_passes_source_agent_to_service() -> None:
    svc = _make_task_service()

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import create_task
        from baize.task.models import TaskSource

        await create_task(title="Buy milk", state=_state())

    call_kwargs = svc.create_task.call_args.kwargs
    assert call_kwargs.get("source") == TaskSource.agent


@pytest.mark.asyncio
async def test_create_task_uses_default_priority_medium() -> None:
    svc = _make_task_service()

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import create_task

        await create_task(title="Some task", state=_state())

    call_kwargs = svc.create_task.call_args.kwargs
    assert call_kwargs.get("priority") == "medium"


@pytest.mark.asyncio
async def test_create_task_passes_due_date_and_priority() -> None:
    svc = _make_task_service()

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import create_task

        await create_task(title="Buy milk", due_date="2026-03-25", priority="high", state=_state())

    call_kwargs = svc.create_task.call_args.kwargs
    assert call_kwargs.get("due_date") == "2026-03-25"
    assert call_kwargs.get("priority") == "high"


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tasks_returns_formatted_list_on_success() -> None:
    tasks = [
        FakeTask(title="Buy groceries"),
        FakeTask(title="Write report"),
    ]
    svc = _make_task_service(list_return=tasks)

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import list_tasks

        result = await list_tasks(state=_state(), status="todo")

    assert "Buy groceries" in result
    assert "Write report" in result


@pytest.mark.asyncio
async def test_list_tasks_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.task._get_task_service", return_value=None):
        from baize.agent.tools.task import list_tasks

        result = await list_tasks(state=_state())

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_list_tasks_handles_empty_results() -> None:
    svc = _make_task_service(list_return=[])

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import list_tasks

        result = await list_tasks(state=_state(), status="todo")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_list_tasks_passes_status_to_service() -> None:
    svc = _make_task_service(list_return=[])

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import list_tasks

        await list_tasks(state=_state(), status="done")

    svc.list_tasks.assert_called_once()
    _, kwargs = svc.list_tasks.call_args
    assert kwargs.get("status") == "done"


@pytest.mark.asyncio
async def test_list_tasks_passes_none_status_when_all() -> None:
    svc = _make_task_service(list_return=[])

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import list_tasks

        await list_tasks(state=_state(), status="all")

    _, kwargs = svc.list_tasks.call_args
    assert kwargs.get("status") is None


# ---------------------------------------------------------------------------
# complete_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_task_returns_confirmation_when_unique_match() -> None:
    task = FakeTask(title="Buy milk")
    svc = _make_task_service(search_return=[task], mark_done_return=task)

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import complete_task

        result = await complete_task(task_title="Buy milk", state=_state())

    assert "Buy milk" in result
    assert "完成" in result


@pytest.mark.asyncio
async def test_complete_task_returns_candidate_list_when_multiple_matches() -> None:
    tasks = [FakeTask(title="Buy milk"), FakeTask(title="Buy coffee")]
    svc = _make_task_service(search_return=tasks)

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import complete_task

        result = await complete_task(task_title="Buy", state=_state())

    assert "Buy milk" in result
    assert "Buy coffee" in result
    svc.mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_complete_task_returns_not_found_when_no_match() -> None:
    svc = _make_task_service(search_return=[])

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import complete_task

        result = await complete_task(task_title="nonexistent task", state=_state())

    assert "未找到" in result
    assert "nonexistent task" in result
    svc.mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_complete_task_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.task._get_task_service", return_value=None):
        from baize.agent.tools.task import complete_task

        result = await complete_task(task_title="some task", state=_state())

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_complete_task_searches_with_task_title() -> None:
    task = FakeTask(title="Deploy service")
    svc = _make_task_service(search_return=[task], mark_done_return=task)
    user_id = str(uuid.uuid4())

    with patch("baize.agent.tools.task._get_task_service", return_value=svc):
        from baize.agent.tools.task import complete_task

        await complete_task(task_title="Deploy", state={"user_id": user_id})

    svc.search_todo_by_title.assert_called_once()
    _, kwargs = svc.search_todo_by_title.call_args
    assert kwargs.get("title") == "Deploy"


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
