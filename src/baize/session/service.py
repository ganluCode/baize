"""Session service: business logic for session and message management."""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from baize.session.models import ChatMessageModel, MessageRole, SessionModel
from baize.session.repository import ChatMessageRepository, SessionRepository

logger = logging.getLogger(__name__)


class SessionService:
    """Orchestrates session lifecycle and message persistence."""

    def __init__(
        self,
        session_repo: SessionRepository,
        message_repo: ChatMessageRepository,
    ) -> None:
        self._session_repo = session_repo
        self._message_repo = message_repo

    async def save_message(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        role: MessageRole,
        content: str,
        tool_calls: dict | None = None,
        tool_name: str | None = None,
        token_usage: dict | None = None,
    ) -> ChatMessageModel:
        """Persist a chat message and bump the parent session's updated_at.

        Args:
            session_id: The session this message belongs to.
            user_id: The user who owns this message.
            role: Message role (user / assistant / system / tool).
            content: Message text.
            tool_calls: Optional tool-call payload.
            tool_name: Optional tool name for tool-role messages.
            token_usage: Optional token usage metadata.

        Returns:
            The persisted ChatMessageModel.
        """
        msg = await self._message_repo.create(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_name=tool_name,
            token_usage=token_usage,
        )
        await self._session_repo.update(session_id, updated_at=datetime.now(timezone.utc))
        logger.debug("Saved message %s for session %s.", msg.id, session_id)
        return msg

    async def get_history(
        self,
        session_id: uuid.UUID,
        limit: int = 50,
    ) -> list[ChatMessageModel]:
        """Return recent messages for a session in ascending order.

        Args:
            session_id: The session to retrieve history for.
            limit: Maximum number of messages to return.

        Returns:
            List of ChatMessageModel ordered by created_at ascending.
        """
        return await self._message_repo.get_recent(session_id, limit)

    async def create_session(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        title: str | None = None,
    ) -> SessionModel:
        """Create a new session for a given user and agent.

        Args:
            user_id: The user who owns the session.
            agent_id: The agent this session belongs to.
            title: Optional session title.

        Returns:
            The newly created SessionModel.
        """
        session = await self._session_repo.create(user_id=user_id, agent_id=agent_id, title=title)
        logger.debug("Created session %s for user %s / agent %s.", session.id, user_id, agent_id)
        return session

    async def list_sessions(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[SessionModel], int]:
        """Return paginated sessions for a user and agent, ordered by updated_at desc.

        Args:
            user_id: Only return sessions belonging to this user.
            agent_id: Only return sessions for this agent.
            status: Optional status filter (SessionStatus value).
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            A tuple of (items, total_count).
        """
        from baize.session.models import SessionStatus

        status_enum = SessionStatus(status) if status is not None else None
        return await self._session_repo.list_by_agent(
            user_id=user_id,
            agent_id=agent_id,
            status=status_enum,
            limit=limit,
            offset=offset,
        )

    async def get_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> SessionModel:
        """Return a session by id, enforcing ownership.

        Args:
            session_id: The session to retrieve.
            user_id: The requesting user's id.

        Returns:
            The SessionModel if found and owned by the user.

        Raises:
            HTTPException: 404 if not found, 403 if not owned by the user.
        """
        session = await self._session_repo.get_by_id(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden.")
        return session

    async def archive_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> SessionModel:
        """Archive a session, enforcing ownership.

        Args:
            session_id: The session to archive.
            user_id: The requesting user's id.

        Returns:
            The updated SessionModel with status=archived.

        Raises:
            HTTPException: 404 if not found, 403 if not owned by the user.
        """
        from baize.session.models import SessionStatus

        await self.get_session(session_id, user_id)
        updated = await self._session_repo.update(session_id, status=SessionStatus.archived)
        assert updated is not None
        logger.debug("Archived session %s.", session_id)
        return updated

    async def delete_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Delete a session, enforcing ownership.

        Args:
            session_id: The session to delete.
            user_id: The requesting user's id.

        Raises:
            HTTPException: 404 if not found, 403 if not owned by the user.
        """
        await self.get_session(session_id, user_id)
        await self._session_repo.delete(session_id)
        logger.debug("Deleted session %s.", session_id)

    async def get_or_create_session(
        self,
        session_id: uuid.UUID | None,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
    ) -> SessionModel:
        """Retrieve an existing session or create a new one.

        Args:
            session_id: Existing session id, or None to create a new session.
            user_id: User id to associate with a newly created session.
            agent_id: Agent id to associate with a newly created session.

        Returns:
            The existing or newly created SessionModel.

        Raises:
            HTTPException: 404 if session_id is provided but the session does not exist.
        """
        if session_id is None:
            session = await self._session_repo.create(user_id=user_id, agent_id=agent_id)
            logger.debug("Created new session %s for user %s.", session.id, user_id)
            return session

        session = await self._session_repo.get_by_id(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session
