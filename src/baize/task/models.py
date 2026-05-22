"""SQLAlchemy async ORM models for tasks table."""

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from baize.user.models import Base


class TaskPriority(enum.StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class TaskStatus(enum.StrEnum):
    todo = "todo"
    in_progress = "in_progress"
    done = "done"


class TaskSource(enum.StrEnum):
    manual = "manual"
    agent = "agent"


class Task(Base):
    """ORM model mapping to the tasks table."""

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    priority: Mapped[TaskPriority] = mapped_column(
        Enum(TaskPriority, name="task_priority"),
        nullable=False,
        default=TaskPriority.medium,
        server_default=TaskPriority.medium.value,
    )
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, name="task_status"),
        nullable=False,
        default=TaskStatus.todo,
        server_default=TaskStatus.todo.value,
    )
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[TaskSource] = mapped_column(
        Enum(TaskSource, name="task_source"),
        nullable=False,
        default=TaskSource.manual,
        server_default=TaskSource.manual.value,
    )
    tags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    related_memory_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_tasks_user_status_created", "user_id", "status", "created_at"),
        Index("idx_tasks_user_due_date", "user_id", "due_date"),
    )
