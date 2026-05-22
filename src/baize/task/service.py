"""Task service: business logic for task management."""

import logging
import uuid
from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from baize.task.models import Task, TaskSource, TaskStatus
from baize.task.repository import TaskRepository
from baize.task.schemas import TaskCreate, TaskListQuery, TaskResponse, TaskUpdate

logger = logging.getLogger(__name__)


class TaskService:
    """Orchestrates task business logic for HTTP request handlers.

    This service is instantiated per-request with an injected TaskRepository.
    """

    def __init__(self, repository: TaskRepository) -> None:
        self._repo = repository

    async def create_task(self, user_id: uuid.UUID, data: TaskCreate) -> TaskResponse:
        """Create a new task, forcing source=manual for API-created tasks.

        Args:
            user_id: The owner of the task.
            data: Task creation data from the API request.

        Returns:
            TaskResponse for the newly created task.
        """
        data.source = TaskSource.manual
        task = await self._repo.create(user_id=user_id, data=data)
        logger.debug("Created task %s for user %s.", task.id, user_id)
        return TaskResponse.model_validate(task)

    async def list_tasks(
        self,
        user_id: uuid.UUID,
        query: TaskListQuery,
    ) -> tuple[list[TaskResponse], int]:
        """Return paginated tasks for a user with optional filters.

        Args:
            user_id: Only return tasks belonging to this user.
            query: Filters and pagination parameters.

        Returns:
            A tuple of (items, total_count).
        """
        tasks, total = await self._repo.list_tasks(user_id=user_id, query=query)
        return [TaskResponse.model_validate(t) for t in tasks], total

    async def get_task(self, user_id: uuid.UUID, task_id: uuid.UUID) -> TaskResponse:
        """Return a task by id, enforcing ownership.

        Args:
            user_id: The requesting user's id.
            task_id: The task to retrieve.

        Returns:
            TaskResponse if found and owned by the user.

        Raises:
            HTTPException: 404 if not found, 403 if not owned by the user.
        """
        task = await self._repo.get_by_id_any(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        if task.user_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden.")
        return TaskResponse.model_validate(task)

    async def update_task(
        self,
        user_id: uuid.UUID,
        task_id: uuid.UUID,
        data: TaskUpdate,
    ) -> TaskResponse:
        """Update a task, enforcing ownership and handling status transitions.

        When status changes to done, completed_at is set to the current UTC time.
        When status changes from done to todo/in_progress, completed_at is cleared.

        Args:
            user_id: The requesting user's id.
            task_id: The task to update.
            data: Fields to update.

        Returns:
            Updated TaskResponse.

        Raises:
            HTTPException: 404 if not found, 403 if not owned by the user.
        """
        task = await self._repo.get_by_id_any(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        if task.user_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden.")

        # Handle completed_at transitions
        if data.status is not None:
            if data.status == TaskStatus.done and task.status != TaskStatus.done:
                task.completed_at = datetime.now(UTC)
            elif data.status != TaskStatus.done and task.status == TaskStatus.done:
                task.completed_at = None

        updated = await self._repo.update(task=task, data=data)
        logger.debug("Updated task %s.", task_id)
        return TaskResponse.model_validate(updated)

    async def delete_task(self, user_id: uuid.UUID, task_id: uuid.UUID) -> None:
        """Delete a task, enforcing ownership.

        Args:
            user_id: The requesting user's id.
            task_id: The task to delete.

        Raises:
            HTTPException: 404 if not found, 403 if not owned by the user.
        """
        task = await self._repo.get_by_id_any(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        if task.user_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden.")
        await self._repo.delete(task)
        logger.debug("Deleted task %s.", task_id)


class TaskEngineService:
    """Session-managing TaskService wrapper for use in agent tools (singleton in container).

    Creates a fresh DB session per call so agent tools can use the task service
    without a per-request FastAPI dependency injection context.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    def _make_service(self, session) -> TaskService:
        return TaskService(TaskRepository(session))

    async def create_task(
        self,
        user_id: uuid.UUID,
        title: str,
        due_date: str | None = None,
        priority: str = "medium",
        source: TaskSource = TaskSource.agent,
    ) -> Task:
        """Create a task and return the ORM Task object.

        Args:
            user_id: The owner of the task.
            title: Task title.
            due_date: Optional ISO date string (YYYY-MM-DD).
            priority: Priority level.
            source: Task source (defaults to agent for tool-created tasks).

        Returns:
            The persisted Task ORM object.
        """
        from datetime import date

        parsed_due_date: date | None = None
        if due_date:
            try:
                parsed_due_date = date.fromisoformat(due_date)
            except ValueError:
                logger.warning("Invalid due_date format '%s'; ignoring.", due_date)

        from baize.task.models import TaskPriority

        try:
            priority_enum = TaskPriority(priority)
        except ValueError:
            priority_enum = TaskPriority.medium

        async with self._session_factory() as session:
            repo = TaskRepository(session)
            data = TaskCreate(title=title, due_date=parsed_due_date, priority=priority_enum, source=source)
            task = await repo.create(user_id=user_id, data=data)
            return task

    async def list_tasks(
        self,
        user_id: uuid.UUID,
        status: str | None = "todo",
    ) -> list[Task]:
        """Return tasks for a user, optionally filtered by status.

        Args:
            user_id: Filter tasks by this user.
            status: Status string ('todo', 'in_progress', 'done') or None/'all' for no filter.

        Returns:
            List of Task ORM objects.
        """
        from baize.task.models import TaskStatus

        status_filter: TaskStatus | None = None
        if status and status != "all":
            try:
                status_filter = TaskStatus(status)
            except ValueError:
                status_filter = None

        async with self._session_factory() as session:
            repo = TaskRepository(session)
            query = TaskListQuery(status=status_filter, limit=50, offset=0)
            tasks, _ = await repo.list_tasks(user_id=user_id, query=query)
            return tasks

    async def search_todo_by_title(self, user_id: uuid.UUID, title: str) -> list[Task]:
        """Search todo tasks by title (ILIKE %title%) for a user.

        Args:
            user_id: Filter tasks by this user.
            title: Substring to match (case-insensitive).

        Returns:
            List of matching Task ORM objects with status=todo.
        """
        async with self._session_factory() as session:
            repo = TaskRepository(session)
            return await repo.search_by_title(user_id=user_id, title=title)

    async def mark_done(self, user_id: uuid.UUID, task_id: uuid.UUID) -> Task:
        """Mark a task as done and set completed_at.

        Args:
            user_id: The task owner (for ownership check).
            task_id: The task to mark as done.

        Returns:
            The updated Task ORM object.

        Raises:
            HTTPException: 404 if not found, 403 if not owned by the user.
        """
        async with self._session_factory() as session:
            repo = TaskRepository(session)
            task = await repo.get_by_id_any(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="Task not found.")
            if task.user_id != user_id:
                raise HTTPException(status_code=403, detail="Forbidden.")
            task.completed_at = datetime.now(UTC)
            data = TaskUpdate(status=TaskStatus.done)
            updated = await repo.update(task=task, data=data)
            return updated
