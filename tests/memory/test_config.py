"""Tests for resolve_memory_config() (F-006).

Acceptance criteria:
- session value wins over all others
- session=None falls through to agent value
- all None falls through to global default
- user.preferences missing key does not raise
"""

from unittest.mock import MagicMock

from baize.memory.config import MemoryConfig
from baize.memory.service import ResolvedMemoryConfig, resolve_memory_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(auto_memory_recall=None, shared_memory=None):
    s = MagicMock()
    s.auto_memory_recall = auto_memory_recall
    s.shared_memory = shared_memory
    return s


def _make_agent(auto_memory_recall=None, shared_memory=None):
    a = MagicMock()
    a.auto_memory_recall = auto_memory_recall
    a.shared_memory = shared_memory
    return a


def _make_user(preferences=None):
    u = MagicMock()
    u.preferences = preferences
    return u


def _make_global(auto_recall=True, shared=True):
    g = MagicMock()
    g.memory = MemoryConfig(auto_recall=auto_recall, shared=shared)
    return g


# ---------------------------------------------------------------------------
# ResolvedMemoryConfig
# ---------------------------------------------------------------------------


class TestResolvedMemoryConfig:
    def test_is_pydantic_or_dataclass(self):
        result = ResolvedMemoryConfig(auto_memory_recall=True, shared_memory=False)
        assert result.auto_memory_recall is True
        assert result.shared_memory is False


# ---------------------------------------------------------------------------
# Session level (highest priority)
# ---------------------------------------------------------------------------


class TestSessionLevel:
    def test_session_values_win_over_agent_and_global(self):
        session = _make_session(auto_memory_recall=False, shared_memory=False)
        agent = _make_agent(auto_memory_recall=True, shared_memory=True)
        user = _make_user(preferences={"auto_memory_recall": True, "shared_memory": True})
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(session, agent, user, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is False

    def test_session_partial_auto_memory_recall_only(self):
        """session sets auto_memory_recall; shared_memory falls through to agent."""
        session = _make_session(auto_memory_recall=False, shared_memory=None)
        agent = _make_agent(auto_memory_recall=True, shared_memory=True)
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(session, agent, None, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is True  # from agent

    def test_session_partial_shared_memory_only(self):
        """session sets shared_memory; auto_memory_recall falls through to agent."""
        session = _make_session(auto_memory_recall=None, shared_memory=False)
        agent = _make_agent(auto_memory_recall=True, shared_memory=True)
        global_cfg = _make_global(auto_recall=False, shared=True)

        result = resolve_memory_config(session, agent, None, global_cfg)

        assert result.auto_memory_recall is True  # from agent
        assert result.shared_memory is False  # from session


# ---------------------------------------------------------------------------
# Agent level (second priority)
# ---------------------------------------------------------------------------


class TestAgentLevel:
    def test_agent_values_used_when_session_is_none(self):
        agent = _make_agent(auto_memory_recall=False, shared_memory=False)
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(None, agent, None, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is False

    def test_agent_values_used_when_session_fields_are_none(self):
        session = _make_session(auto_memory_recall=None, shared_memory=None)
        agent = _make_agent(auto_memory_recall=False, shared_memory=True)
        global_cfg = _make_global(auto_recall=True, shared=False)

        result = resolve_memory_config(session, agent, None, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is True


# ---------------------------------------------------------------------------
# User level (third priority)
# ---------------------------------------------------------------------------


class TestUserLevel:
    def test_user_preferences_used_when_session_and_agent_are_none(self):
        user = _make_user(preferences={"auto_memory_recall": False, "shared_memory": False})
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(None, None, user, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is False

    def test_user_preferences_missing_key_does_not_raise(self):
        """Missing keys in preferences dict fall through gracefully."""
        user = _make_user(preferences={})
        global_cfg = _make_global(auto_recall=True, shared=False)

        result = resolve_memory_config(None, None, user, global_cfg)

        assert result.auto_memory_recall is True  # global fallback
        assert result.shared_memory is False

    def test_user_preferences_none_falls_through(self):
        """preferences=None falls through to global."""
        user = _make_user(preferences=None)
        global_cfg = _make_global(auto_recall=False, shared=True)

        result = resolve_memory_config(None, None, user, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is True

    def test_user_partial_preference_one_key(self):
        """Only auto_memory_recall in prefs; shared_memory falls to global."""
        user = _make_user(preferences={"auto_memory_recall": False})
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(None, None, user, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is True  # global


# ---------------------------------------------------------------------------
# Global level (lowest priority / fallback)
# ---------------------------------------------------------------------------


class TestGlobalLevel:
    def test_global_defaults_used_when_all_upper_levels_are_none(self):
        global_cfg = _make_global(auto_recall=False, shared=False)

        result = resolve_memory_config(None, None, None, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is False

    def test_global_defaults_true(self):
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(None, None, None, global_cfg)

        assert result.auto_memory_recall is True
        assert result.shared_memory is True
