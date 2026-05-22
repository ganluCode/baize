"""Context engineering module for Baize.

Encapsulates all logic for preparing the LLM input context:
- System prompt layered assembly (agent prompt / user prefs / tools / memories)
- Conversation history compression
- Memory strategy resolution
- Memory recall orchestration

Exposed entry point: :class:`ContextService` via :func:`prepare`.
"""

from baize.context.schemas import PreparedContext
from baize.context.service import ContextService

__all__ = ["ContextService", "PreparedContext"]
