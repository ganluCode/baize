"""Abstract interface for the Baize memory service.

Concrete adapters (Mem0, EverMemOS, …) implement this interface so that
upper layers can remain decoupled from a specific memory backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """A single memory record returned by the memory service."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None  # relevance score from semantic search, if available


class MemoryServiceInterface(ABC):
    """Abstract base class for memory service implementations."""

    @abstractmethod
    async def add_memory(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Persist a new memory and return its unique ID.

        Args:
            content: The text content to store.
            metadata: Optional key-value metadata attached to the memory.

        Returns:
            The unique identifier of the stored memory.
        """

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Memory]:
        """Search for memories semantically similar to *query*.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters to narrow results.

        Returns:
            List of matching :class:`Memory` objects, ordered by relevance.
        """

    @abstractmethod
    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by its ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The :class:`Memory` object, or *None* if not found.
        """

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by its ID.

        Args:
            memory_id: The unique identifier of the memory to delete.

        Returns:
            *True* if the memory was deleted, *False* if it was not found.
        """
