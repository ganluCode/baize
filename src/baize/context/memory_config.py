"""Memory configuration resolution.

Provides :class:`ResolvedMemoryConfig` and :func:`resolve_memory_config` which
merge per-agent memory settings with global defaults.

Priority chain (highest first):
    agent_config.memory_config > global_config.memory
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from baize.agent.models import AgentConfig
    from baize.core.config import Settings


class ResolvedMemoryConfig(BaseModel):
    """Effective memory configuration after resolution."""

    auto_memory_recall: bool
    shared_memory: bool
    top_k: int = 5


def resolve_memory_config(
    agent_config: AgentConfig | None,
    global_config: Settings,
) -> ResolvedMemoryConfig:
    """Resolve effective memory flags using a two-level priority chain.

    Priority: agent_config.memory_config > global default.

    Args:
        agent_config: Agent configuration ORM object, or ``None``.
        global_config: Application-wide :class:`~baize.core.config.Settings`.

    Returns:
        :class:`ResolvedMemoryConfig` with concrete values.
    """
    # Defaults from global config
    auto_recall = global_config.memory.auto_recall
    shared = global_config.memory.shared
    top_k = 5

    # Agent-level override
    if agent_config is not None and agent_config.memory_config:
        mc = agent_config.memory_config
        if isinstance(mc, dict):
            if mc.get("auto_recall") is not None:
                auto_recall = bool(mc["auto_recall"])
            if mc.get("shared") is not None:
                shared = bool(mc["shared"])
            if mc.get("top_k") is not None:
                top_k = int(mc["top_k"])

    return ResolvedMemoryConfig(
        auto_memory_recall=auto_recall,
        shared_memory=shared,
        top_k=top_k,
    )
