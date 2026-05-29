"""Unit tests for graph registration (F-014).

Verifies that importing baize.agent.graphs registers both "chat" and "knowledge"
builders in the global registry.
"""

from __future__ import annotations

import pytest

import baize.agent.graphs  # noqa: F401 — triggers @register_graph side-effects
from baize.agent.graphs.base import get_graph_builder, list_agent_types


class TestGraphRegistration:
    """After importing baize.agent.graphs, both 'chat' and 'knowledge' must be registered."""

    def test_knowledge_registered(self):
        """Registry contains 'knowledge' key after import."""
        assert "knowledge" in list_agent_types()

    def test_chat_still_registered(self):
        """Registry still contains 'chat' key (not overwritten by knowledge import)."""
        assert "chat" in list_agent_types()

    def test_both_types_registered(self):
        """Both 'chat' and 'knowledge' are present simultaneously."""
        types = list_agent_types()
        assert "chat" in types
        assert "knowledge" in types

    def test_get_graph_builder_knowledge_does_not_raise(self):
        """get_graph_builder('knowledge') returns a builder without raising."""
        builder = get_graph_builder("knowledge")
        assert builder is not None

    def test_get_graph_builder_chat_does_not_raise(self):
        """get_graph_builder('chat') returns a builder without raising."""
        builder = get_graph_builder("chat")
        assert builder is not None

    def test_knowledge_builder_has_correct_agent_type(self):
        """Knowledge builder's agent_type attribute is 'knowledge'."""
        builder = get_graph_builder("knowledge")
        assert builder.agent_type == "knowledge"

    def test_chat_builder_has_correct_agent_type(self):
        """Chat builder's agent_type attribute is 'chat'."""
        builder = get_graph_builder("chat")
        assert builder.agent_type == "chat"

    def test_unknown_agent_type_raises_value_error(self):
        """get_graph_builder raises ValueError for unregistered agent types."""
        with pytest.raises(ValueError, match="Unknown agent_type"):
            get_graph_builder("nonexistent_type")
