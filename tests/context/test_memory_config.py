"""Tests for resolve_memory_config() — 2-level priority (agent > global)."""

from unittest.mock import MagicMock

from baize.context.memory_config import ResolvedMemoryConfig, resolve_memory_config
from baize.memory.config import MemoryConfig


def _make_agent(memory_config: dict | None = None):
    a = MagicMock()
    a.memory_config = memory_config
    return a


def _make_global(auto_recall: bool = True, shared: bool = True):
    g = MagicMock()
    g.memory = MemoryConfig(auto_recall=auto_recall, shared=shared)
    return g


class TestResolvedMemoryConfig:
    def test_construction(self):
        result = ResolvedMemoryConfig(auto_memory_recall=True, shared_memory=False)
        assert result.auto_memory_recall is True
        assert result.shared_memory is False
        assert result.top_k == 5  # default


class TestAgentLevel:
    def test_agent_values_override_global(self):
        agent = _make_agent({"auto_recall": False, "shared": False, "top_k": 10})
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(agent, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is False
        assert result.top_k == 10

    def test_agent_partial_override_auto_recall_only(self):
        """agent sets only auto_recall; shared falls through to global."""
        agent = _make_agent({"auto_recall": False})
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(agent, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is True  # from global

    def test_agent_partial_override_shared_only(self):
        agent = _make_agent({"shared": False})
        global_cfg = _make_global(auto_recall=False, shared=True)

        result = resolve_memory_config(agent, global_cfg)

        assert result.auto_memory_recall is False  # from global
        assert result.shared_memory is False

    def test_agent_empty_memory_config_falls_through(self):
        agent = _make_agent({})
        global_cfg = _make_global(auto_recall=True, shared=False)

        result = resolve_memory_config(agent, global_cfg)

        assert result.auto_memory_recall is True
        assert result.shared_memory is False

    def test_agent_memory_config_none_falls_through(self):
        agent = _make_agent(None)
        global_cfg = _make_global(auto_recall=False, shared=True)

        result = resolve_memory_config(agent, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is True


class TestGlobalLevel:
    def test_global_defaults_when_agent_is_none(self):
        global_cfg = _make_global(auto_recall=False, shared=False)

        result = resolve_memory_config(None, global_cfg)

        assert result.auto_memory_recall is False
        assert result.shared_memory is False

    def test_global_defaults_true(self):
        global_cfg = _make_global(auto_recall=True, shared=True)

        result = resolve_memory_config(None, global_cfg)

        assert result.auto_memory_recall is True
        assert result.shared_memory is True
        assert result.top_k == 5
