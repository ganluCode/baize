"""Task router: REST API endpoints for task management."""

import logging
import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from baize.core.database import get_db
from baize.task.models import TaskPriority, TaskStatus
from baize.task.repository import TaskRepository
from baize.task.schemas import (
    TaskCreate,
    TaskListQuery,
    TaskListResponse,
    TaskResponse,
    TaskUpdate,
)
from baize.task.service import TaskService
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _get_task_service(db: AsyncSession = Depends(get_db)) -> TaskService:
    """Dependency factory for TaskService — builds a per-request instance."""
    return TaskService(TaskRepository(db))


@router.post(
    "",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_task(
    body: TaskCreate,
    current_user: UserModel = Depends(get_current_user),
    svc: TaskService = Depends(_get_task_service),
) -> TaskResponse:
    """Create a new task for the authenticated user.

    Returns:
        TaskResponse with status 201. source is always set to 'manual'.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    return await svc.create_task(user_id=current_user.id, data=body)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: TaskStatus | None = Query(default=None),
    priority: TaskPriority | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserModel = Depends(get_current_user),
    svc: TaskService = Depends(_get_task_service),
) -> TaskListResponse:
    """Return paginated tasks for the authenticated user.

    Returns:
        TaskListResponse with items and total count.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    query = TaskListQuery(status=status, priority=priority, limit=limit, offset=offset)
    items, total = await svc.list_tasks(user_id=current_user.id, query=query)
    return TaskListResponse(items=items, total=total)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: TaskService = Depends(_get_task_service),
) -> TaskResponse:
    """Return a single task owned by the authenticated user.

    Returns:
        TaskResponse with status 200.

    Raises:
        HTTPException: 401 if not authenticated, 404 if not found, 403 if not owner.
    """
    return await svc.get_task(user_id=current_user.id, task_id=task_id)


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: uuid.UUID,
    body: TaskUpdate,
    current_user: UserModel = Depends(get_current_user),
    svc: TaskService = Depends(_get_task_service),
) -> TaskResponse:
    """Partially update a task owned by the authenticated user.

    When status is set to 'done', completed_at is automatically populated.

    Returns:
        Updated TaskResponse.

    Raises:
        HTTPException: 401 if not authenticated, 404 if not found, 403 if not owner.
    """
    return await svc.update_task(user_id=current_user.id, task_id=task_id, data=body)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: uuid.UUID,
    current_user: UserModel = Depends(get_current_user),
    svc: TaskService = Depends(_get_task_service),
) -> None:
    """Delete a task owned by the authenticated user.

    Returns:
        204 No Content.

    Raises:
        HTTPException: 401 if not authenticated, 404 if not found, 403 if not owner.
    """
    await svc.delete_task(user_id=current_user.id, task_id=task_id)
