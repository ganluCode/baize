"""Pydantic v2 request/response schemas for the task module."""

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field

from baize.task.models import TaskPriority, TaskSource, TaskStatus


class TaskCreate(BaseModel):
    """Request body for creating a new task."""

    title: str = Field(..., min_length=1, max_length=200)
    content: str | None = None
    priority: TaskPriority = TaskPriority.medium
    due_date: date | None = None
    tags: list[str] | None = None
    source: TaskSource = TaskSource.manual


class TaskUpdate(BaseModel):
    """Request body for partially updating a task (PATCH)."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = None
    priority: TaskPriority | None = None
    status: TaskStatus | None = None
    due_date: date | None = None
    tags: list[str] | None = None

    model_config = {"extra": "ignore"}


class TaskResponse(BaseModel):
    """Task representation returned by the API."""

    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    content: str | None
    priority: TaskPriority
    status: TaskStatus
    due_date: date | None
    source: TaskSource
    tags: list | None
    related_memory_id: str | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    """Paginated list of tasks."""

    items: list[TaskResponse]
    total: int


class TaskListQuery(BaseModel):
    """Query parameters for listing tasks."""

    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    tags: list[str] | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
