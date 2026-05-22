"""Repository for Task database operations."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.task.models import Task, TaskStatus
from baize.task.schemas import TaskCreate, TaskListQuery, TaskUpdate


class TaskRepository:
    """Data access layer for Task model."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: uuid.UUID, data: TaskCreate) -> Task:
        """Insert a new task and return the persisted Task.

        Args:
            user_id: The owner of the task.
            data: Task creation data.

        Returns:
            The persisted Task object.
        """
        obj = Task(
            user_id=user_id,
            title=data.title,
            content=data.content,
            priority=data.priority,
            status=TaskStatus.todo,
            due_date=data.due_date,
            source=data.source,
            tags=data.tags,
        )
        self._session.add(obj)
        await self._session.commit()
        await self._session.refresh(obj)
        return obj

    async def get_by_id(self, task_id: uuid.UUID, user_id: uuid.UUID) -> Task | None:
        """Return the task with the given id owned by user_id, or None.

        Returns None if the task does not exist OR if it belongs to a different user.
        Callers that need to distinguish 404 vs 403 should use a two-step lookup.

        Args:
            task_id: The task id to look up.
            user_id: The requesting user's id.

        Returns:
            Task if found and owned, otherwise None.
        """
        result = await self._session.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        if task is None:
            return None
        if task.user_id != user_id:
            # Return sentinel: task exists but wrong owner.
            # Callers use get_by_id_any to distinguish 403 vs 404.
            return None
        return task

    async def get_by_id_any(self, task_id: uuid.UUID) -> Task | None:
        """Return the task by id regardless of owner, or None if not found.

        Args:
            task_id: The task id to look up.

        Returns:
            Task if found, None if not found.
        """
        result = await self._session.execute(
            select(Task).where(Task.id == task_id)
        )
        return result.scalar_one_or_none()

    async def list_tasks(
        self,
        user_id: uuid.UUID,
        query: TaskListQuery,
    ) -> tuple[list[Task], int]:
        """Return paginated tasks for a user with optional filters.

        Sorting: tasks with a due_date come first (ascending), then tasks
        without a due_date (descending by created_at).

        Args:
            user_id: Only return tasks belonging to this user.
            query: Filters and pagination parameters.

        Returns:
            A tuple of (items, total_count).
        """
        base = select(Task).where(Task.user_id == user_id)

        if query.status is not None:
            base = base.where(Task.status == query.status)
        if query.priority is not None:
            base = base.where(Task.priority == query.priority)
        if query.tags:
            # JSONB @> operator: task.tags must contain all specified tags
            base = base.where(Task.tags.cast(type_=None).op("@>")(query.tags))  # type: ignore[attr-defined]

        count_result = await self._session.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar_one()

        # due_date ASC NULLS LAST, then created_at DESC
        items_result = await self._session.execute(
            base.order_by(
                Task.due_date.asc().nulls_last(),
                Task.created_at.desc(),
            )
            .offset(query.offset)
            .limit(query.limit)
        )
        items = list(items_result.scalars().all())
        return items, total

    async def update(self, task: Task, data: TaskUpdate) -> Task:
        """Apply non-None fields from data to task and persist.

        Args:
            task: The Task ORM object to update.
            data: Fields to update; None values are skipped.

        Returns:
            The updated and refreshed Task.
        """
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(task, key, value)
        await self._session.commit()
        await self._session.refresh(task)
        return task

    async def delete(self, task: Task) -> None:
        """Delete the task record.

        Args:
            task: The Task ORM object to delete.
        """
        await self._session.delete(task)
        await self._session.commit()

    async def search_by_title(
        self,
        user_id: uuid.UUID,
        title: str,
        status: TaskStatus = TaskStatus.todo,
    ) -> list[Task]:
        """Return tasks whose title matches ILIKE %title% for the given user and status.

        Args:
            user_id: Filter by this user.
            title: Substring to match (case-insensitive).
            status: Task status filter (defaults to todo).

        Returns:
            List of matching Task objects.
        """
        result = await self._session.execute(
            select(Task)
            .where(Task.user_id == user_id)
            .where(Task.status == status)
            .where(Task.title.ilike(f"%{title}%"))
            .order_by(Task.created_at.desc())
        )
        return list(result.scalars().all())
