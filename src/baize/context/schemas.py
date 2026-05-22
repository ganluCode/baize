"""Context preparation output schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from baize.context.memory_config import ResolvedMemoryConfig
    from baize.memory.interface import Memory


@dataclass
class PreparedContext:
    """Result of :meth:`ContextService.prepare`.

    Attributes:
        system_prompt: Fully assembled system prompt (all layers).
        history: Conversation history, already compressed to fit the token budget.
        memories: Memories recalled for this turn (empty when auto-recall disabled).
        resolved_memory: Effective memory configuration used for this turn.
    """

    system_prompt: str
    history: list["BaseMessage"]
    resolved_memory: "ResolvedMemoryConfig"
    memories: list["Memory"] = field(default_factory=list)
