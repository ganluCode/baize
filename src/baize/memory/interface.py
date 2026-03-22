"""Abstract interface for the Baize memory service.

Concrete adapters (Mem0, EverMemOS, …) implement this interface so that
upper layers can remain decoupled from a specific memory backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MemoryItem(BaseModel):
    """A single memory record."""

    id: str
    content: str
    user_id: str
    agent_id: str | None = None       # agent that produced this memory (private memories)
    session_id: str | None = None     # session that produced this memory
    shared: bool = True               # False = visible only to the originating agent_id
    metadata: dict[str, Any] | None = None
    score: float | None = None        # relevance score from semantic search
    created_at: datetime
    updated_at: datetime


# Backward-compat alias used by pre-existing tool tests.
Memory = MemoryItem


class MemoryServiceInterface(ABC):
    """Abstract base class for memory service implementations."""

    @abstractmethod
    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        shared: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory and return its unique ID.

        Args:
            content: Text content to store.
            user_id: Owner of this memory.
            agent_id: Agent that produced the memory (marks it as private when provided
                      and *shared* is False).
            session_id: Session that produced the memory.
            shared: When False the memory is visible only to *agent_id*.
            metadata: Optional key-value metadata.

        Returns:
            The unique identifier of the stored memory.
        """

    @abstractmethod
    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """Semantic search for memories relevant to *query*.

        Returns shared memories plus private memories belonging to *agent_id*
        (when provided). Results are ordered by relevance score descending.

        Args:
            query: Natural-language search query.
            user_id: Restrict results to this user's memories.
            agent_id: Also include private memories for this agent.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`MemoryItem` objects ordered by relevance.
        """

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a single memory by its ID.

        Args:
            memory_id: Unique identifier of the memory.

        Returns:
            The :class:`MemoryItem`, or *None* if not found.
        """

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by its ID.

        Args:
            memory_id: Unique identifier of the memory to delete.

        Returns:
            *True* if deleted, *False* if not found.
        """

    @abstractmethod
    async def list_all(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """List all memories for a user (paginated).

        Args:
            user_id: Restrict results to this user's memories.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of :class:`MemoryItem` objects.
        """
