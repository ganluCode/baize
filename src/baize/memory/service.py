"""Memory service utilities for Baize.

Provides configuration resolution helpers that merge per-session, per-agent,
per-user, and global memory settings according to a defined priority chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from baize.agent.models import AgentConfig
    from baize.core.config import Settings
    from baize.session.models import SessionModel
    from baize.user.models import UserModel


class ResolvedMemoryConfig(BaseModel):
    """Effective memory configuration after priority-chain resolution."""

    auto_memory_recall: bool
    shared_memory: bool


def resolve_memory_config(
    session: "SessionModel | None",
    agent_config: "AgentConfig | None",
    user: "UserModel | None",
    global_config: "Settings",
) -> ResolvedMemoryConfig:
    """Resolve effective memory flags using a four-level priority chain.

    Priority (highest first): session > agent_config > user.preferences > global.

    Each level is skipped when its value is ``None``.  The global level always
    provides a non-``None`` fallback.

    Args:
        session: Current session ORM object, or ``None``.
        agent_config: Agent configuration ORM object, or ``None``.
        user: Authenticated user ORM object, or ``None``.
        global_config: Application-wide :class:`~baize.core.config.Settings`.

    Returns:
        :class:`ResolvedMemoryConfig` with concrete ``bool`` values for both
        ``auto_memory_recall`` and ``shared_memory``.
    """
    auto_memory_recall = _resolve_flag(
        levels=[
            _session_field(session, "auto_memory_recall"),
            _agent_field(agent_config, "auto_memory_recall"),
            _user_pref(user, "auto_memory_recall"),
        ],
        default=global_config.memory.auto_recall,
    )
    shared_memory = _resolve_flag(
        levels=[
            _session_field(session, "shared_memory"),
            _agent_field(agent_config, "shared_memory"),
            _user_pref(user, "shared_memory"),
        ],
        default=global_config.memory.shared,
    )
    return ResolvedMemoryConfig(
        auto_memory_recall=auto_memory_recall,
        shared_memory=shared_memory,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_flag(levels: list[bool | None], default: bool) -> bool:
    for value in levels:
        if value is not None:
            return value
    return default


def _session_field(session: Any, field: str) -> bool | None:
    if session is None:
        return None
    return getattr(session, field, None)


def _agent_field(agent_config: Any, field: str) -> bool | None:
    if agent_config is None:
        return None
    return getattr(agent_config, field, None)


def _user_pref(user: Any, key: str) -> bool | None:
    if user is None:
        return None
    prefs = getattr(user, "preferences", None)
    if not isinstance(prefs, dict):
        return None
    value = prefs.get(key)
    if value is None:
        return None
    return bool(value)
