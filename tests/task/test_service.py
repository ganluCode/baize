"""Tests for TaskService — covers business rules and exception scenarios."""

import uuid
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from baize.task.models import Task, TaskPriority, TaskSource, TaskStatus
from baize.task.repository import TaskRepository
from baize.task.schemas import TaskCreate, TaskListQuery, TaskResponse, TaskUpdate
from baize.task.service import TaskService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    title: str = "Test Task",
    status: TaskStatus = TaskStatus.todo,
    priority: TaskPriority = TaskPriority.medium,
    source: TaskSource = TaskSource.manual,
    due_date: date | None = None,
    completed_at: datetime | None = None,
) -> Task:
    t = Task()
    t.id = task_id or uuid.uuid4()
    t.user_id = user_id or uuid.uuid4()
    t.title = title
    t.content = None
    t.status = status
    t.priority = priority
    t.source = source
    t.due_date = due_date
    t.tags = None
    t.related_memory_id = None
    t.completed_at = completed_at
    t.created_at = datetime.now(timezone.utc)
    t.updated_at = datetime.now(timezone.utc)
    return t


@pytest.fixture
def repo() -> AsyncMock:
    return AsyncMock(spec=TaskRepository)


@pytest.fixture
def service(repo: AsyncMock) -> TaskService:
    return TaskService(repository=repo)


# ---------------------------------------------------------------------------
# create_task — source is forced to manual
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_sets_source_manual(service: TaskService, repo: AsyncMock) -> None:
    """create_task always sets source=manual regardless of input."""
    user_id = uuid.uuid4()
    task = _make_task(user_id=user_id, source=TaskSource.manual)
    repo.create.return_value = task

    data = TaskCreate(title="My Task", source=TaskSource.agent)
    result = await service.create_task(user_id=user_id, data=data)

    # Source in the data passed to repo must be manual
    call_args = repo.create.call_args
    passed_data: TaskCreate = call_args.kwargs["data"]
    assert passed_data.source == TaskSource.manual
    assert isinstance(result, TaskResponse)


@pytest.mark.asyncio
async def test_create_task_returns_task_response(service: TaskService, repo: AsyncMock) -> None:
    user_id = uuid.uuid4()
    task = _make_task(user_id=user_id)
    repo.create.return_value = task

    result = await service.create_task(user_id=user_id, data=TaskCreate(title="Hello"))

    assert result.title == task.title
    assert result.user_id == user_id


# ---------------------------------------------------------------------------
# update_task — completed_at transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_task_fills_completed_at(service: TaskService, repo: AsyncMock) -> None:
    """Updating status to done sets completed_at to current UTC time."""
    user_id = uuid.uuid4()
    task_id = uuid.uuid4()
    task = _make_task(task_id=task_id, user_id=user_id, status=TaskStatus.todo, completed_at=None)
    updated = _make_task(
        task_id=task_id, user_id=user_id, status=TaskStatus.done, completed_at=datetime.now(timezone.utc)
    )
    repo.get_by_id_any.return_value = task
    repo.update.return_value = updated

    result = await service.update_task(user_id=user_id, task_id=task_id, data=TaskUpdate(status=TaskStatus.done))

    # completed_at must have been set on the task ORM object before update
    assert task.completed_at is not None
    assert isinstance(result, TaskResponse)


@pytest.mark.asyncio
async def test_reopen_task_clears_completed_at(service: TaskService, repo: AsyncMock) -> None:
    """Updating status from done to todo clears completed_at."""
    user_id = uuid.uuid4()
    task_id = uuid.uuid4()
    task = _make_task(
        task_id=task_id, user_id=user_id, status=TaskStatus.done, completed_at=datetime.now(timezone.utc)
    )
    updated = _make_task(task_id=task_id, user_id=user_id, status=TaskStatus.todo, completed_at=None)
    repo.get_by_id_any.return_value = task
    repo.update.return_value = updated

    await service.update_task(user_id=user_id, task_id=task_id, data=TaskUpdate(status=TaskStatus.todo))

    # completed_at must have been cleared on the task ORM object before update
    assert task.completed_at is None


@pytest.mark.asyncio
async def test_complete_task_no_change_when_already_done(service: TaskService, repo: AsyncMock) -> None:
    """Updating a task already done to done again does not reset completed_at."""
    user_id = uuid.uuid4()
    task_id = uuid.uuid4()
    original_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    task = _make_task(task_id=task_id, user_id=user_id, status=TaskStatus.done, completed_at=original_ts)
    repo.get_by_id_any.return_value = task
    repo.update.return_value = task

    await service.update_task(user_id=user_id, task_id=task_id, data=TaskUpdate(status=TaskStatus.done))

    # completed_at should remain unchanged
    assert task.completed_at == original_ts


# ---------------------------------------------------------------------------
# get_task — 404 / 403
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_task_not_found_raises_404(service: TaskService, repo: AsyncMock) -> None:
    """get_task raises HTTP 404 when the task does not exist."""
    repo.get_by_id_any.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await service.get_task(user_id=uuid.uuid4(), task_id=uuid.uuid4())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_task_wrong_user_raises_403(service: TaskService, repo: AsyncMock) -> None:
    """get_task raises HTTP 403 when the task exists but belongs to another user."""
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    task = _make_task(user_id=owner_id)
    repo.get_by_id_any.return_value = task

    with pytest.raises(HTTPException) as exc_info:
        await service.get_task(user_id=other_id, task_id=task.id)

    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# list_tasks — user isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tasks_user_isolation(service: TaskService, repo: AsyncMock) -> None:
    """list_tasks only returns tasks belonging to the given user_id."""
    user1 = uuid.uuid4()
    tasks = [_make_task(user_id=user1), _make_task(user_id=user1)]
    repo.list_tasks.return_value = (tasks, 2)

    items, total = await service.list_tasks(user_id=user1, query=TaskListQuery())

    # Repository is called with user1's id
    repo.list_tasks.assert_called_once()
    call_kwargs = repo.list_tasks.call_args.kwargs
    assert call_kwargs["user_id"] == user1
    assert total == 2
    assert all(r.user_id == user1 for r in items)


# ---------------------------------------------------------------------------
# delete_task — 404 / 403
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_task_not_found_raises_404(service: TaskService, repo: AsyncMock) -> None:
    repo.get_by_id_any.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await service.delete_task(user_id=uuid.uuid4(), task_id=uuid.uuid4())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_task_wrong_user_raises_403(service: TaskService, repo: AsyncMock) -> None:
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    task = _make_task(user_id=owner_id)
    repo.get_by_id_any.return_value = task

    with pytest.raises(HTTPException) as exc_info:
        await service.delete_task(user_id=other_id, task_id=task.id)

    assert exc_info.value.status_code == 403
