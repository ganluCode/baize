"""Repository classes for Session and ChatMessage database operations."""

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.session.models import ChatMessageModel, MessageRole, SessionModel, SessionStatus


class SessionRepository:
    """Data access layer for SessionModel."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        title: str | None = None,
    ) -> SessionModel:
        """Insert a new session record and return the persisted SessionModel."""
        obj = SessionModel(user_id=user_id, agent_id=agent_id, title=title)
        self._session.add(obj)
        await self._session.commit()
        await self._session.refresh(obj)
        return obj

    async def get_by_id(self, session_id: uuid.UUID) -> SessionModel | None:
        """Return the session with the given id, or None if not found."""
        result = await self._session.execute(
            select(SessionModel).where(SessionModel.id == session_id)
        )
        return result.scalar_one_or_none()

    async def list_by_agent(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        status: SessionStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[SessionModel], int]:
        """Return paginated sessions for a given user + agent, ordered by updated_at desc.

        Args:
            user_id: Only return sessions belonging to this user.
            agent_id: Only return sessions for this agent.
            status: Optional status filter.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            A tuple of (items, total_count).
        """
        base = (
            select(SessionModel)
            .where(SessionModel.user_id == user_id)
            .where(SessionModel.agent_id == agent_id)
            .where(SessionModel.status != SessionStatus.deleted)
        )
        if status is not None:
            base = base.where(SessionModel.status == status)

        count_result = await self._session.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar_one()

        items_result = await self._session.execute(
            base.order_by(SessionModel.updated_at.desc()).offset(offset).limit(limit)
        )
        items = list(items_result.scalars().all())
        return items, total

    async def update(self, session_id: uuid.UUID, **kwargs) -> SessionModel | None:
        """Update supported fields (status, title) and return the refreshed session.

        Args:
            session_id: The id of the session to update.
            **kwargs: Fields to update — only 'status' and 'title' are applied.

        Returns:
            The updated SessionModel, or None if the session was not found.
        """
        obj = await self.get_by_id(session_id)
        if obj is None:
            return None

        allowed = {"status", "title", "updated_at", "title_gen_attempts"}
        for key, value in kwargs.items():
            if key in allowed:
                setattr(obj, key, value)

        await self._session.commit()
        await self._session.refresh(obj)
        return obj

    async def delete(self, session_id: uuid.UUID) -> None:
        """Delete the session record; chat_messages are removed via CASCADE.

        Args:
            session_id: The id of the session to delete.
        """
        obj = await self.get_by_id(session_id)
        if obj is not None:
            await self._session.delete(obj)
            await self._session.commit()

    async def delete_by_agent_id(self, agent_id: uuid.UUID) -> None:
        """Delete all sessions belonging to the given agent.

        chat_messages are removed via CASCADE on the FK.

        Args:
            agent_id: All sessions with this agent_id will be deleted.
        """
        await self._session.execute(
            delete(SessionModel).where(SessionModel.agent_id == agent_id)
        )
        await self._session.commit()


class ChatMessageRepository:
    """Data access layer for ChatMessageModel."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        role: MessageRole,
        content: str,
        thinking: str | None = None,
        tool_calls: dict | None = None,
        tool_name: str | None = None,
        token_usage: dict | None = None,
    ) -> ChatMessageModel:
        """Insert a new chat message and return the persisted ChatMessageModel."""
        obj = ChatMessageModel(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            tool_name=tool_name,
            token_usage=token_usage,
        )
        self._session.add(obj)
        await self._session.commit()
        await self._session.refresh(obj)
        return obj

    async def list_by_session(
        self,
        session_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        before: uuid.UUID | None = None,
    ) -> tuple[list[ChatMessageModel], int]:
        """Return paginated messages for a session.

        Supports two pagination modes:
        - **before (cursor)**: Return ``limit`` messages older than the given
          message ID, ordered ascending. Used for "load earlier" in chat UI.
        - **offset**: Traditional offset pagination, ordered ascending.

        In both modes the returned list is sorted by ``created_at ASC``
        (oldest first) so the frontend can render top-to-bottom.

        Args:
            session_id: The session to query messages for.
            limit: Maximum number of records to return.
            offset: Number of records to skip (ignored when ``before`` is set).
            before: Cursor — return messages created before this message ID.

        Returns:
            A tuple of (items, total_count).
        """
        base = select(ChatMessageModel).where(ChatMessageModel.session_id == session_id)

        count_result = await self._session.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar_one()

        if before is not None:
            # Cursor-based: get the timestamp of the cursor message
            cursor_result = await self._session.execute(
                select(ChatMessageModel.created_at).where(ChatMessageModel.id == before)
            )
            cursor_ts = cursor_result.scalar_one_or_none()
            if cursor_ts is not None:
                # Get `limit` messages before the cursor, then sort ascending
                query = (
                    base.where(ChatMessageModel.created_at < cursor_ts)
                    .order_by(ChatMessageModel.created_at.desc())
                    .limit(limit)
                )
                items_result = await self._session.execute(query)
                items = list(reversed(items_result.scalars().all()))
            else:
                items = []
        else:
            # No cursor: return the latest `limit` messages (for initial load)
            # Query DESC then reverse to get ASC order
            query = (
                base.order_by(ChatMessageModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            items_result = await self._session.execute(query)
            items = list(reversed(items_result.scalars().all()))

        return items, total

    async def get_recent(
        self,
        session_id: uuid.UUID,
        limit: int,
    ) -> list[ChatMessageModel]:
        """Return the most recent N messages for a session, ordered by created_at asc.

        Args:
            session_id: The session to query messages for.
            limit: Maximum number of records to return.

        Returns:
            A list of ChatMessageModel ordered by created_at ascending.
        """
        # Fetch the latest N rows using a subquery ordered desc, then re-order asc
        subq = (
            select(ChatMessageModel)
            .where(ChatMessageModel.session_id == session_id)
            .order_by(ChatMessageModel.created_at.desc())
            .limit(limit)
            .subquery()
        )
        result = await self._session.execute(
            select(ChatMessageModel)
            .join(subq, ChatMessageModel.id == subq.c.id)
            .order_by(ChatMessageModel.created_at.asc())
        )
        return list(result.scalars().all())
