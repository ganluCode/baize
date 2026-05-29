"""Unit tests for KnowledgeGraph (F-009)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_llm():
    llm = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


def _make_agent_config(*, knowledge_config=None, tools=None):
    config = MagicMock()
    config.knowledge_config = knowledge_config
    config.tools = tools
    return config


def _make_retrieval_svc():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    return svc


_KB_ID = str(uuid.uuid4())


class TestKnowledgeGraphMissingServices:
    """build() raises ValueError when services['retrieval'] is absent."""

    def test_build_raises_when_services_is_none(self):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(
            knowledge_config={"default_kb_id": _KB_ID}
        )

        with pytest.raises(ValueError, match="retrieval"):
            builder.build(
                llm=_make_llm(),
                tools=[],
                agent_config=agent_config,
                services=None,
            )

    def test_build_raises_when_services_missing_retrieval_key(self):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(
            knowledge_config={"default_kb_id": _KB_ID}
        )

        with pytest.raises(ValueError, match="retrieval"):
            builder.build(
                llm=_make_llm(),
                tools=[],
                agent_config=agent_config,
                services={"other": "value"},
            )

    def test_build_raises_when_services_empty_dict(self):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(
            knowledge_config={"default_kb_id": _KB_ID}
        )

        with pytest.raises(ValueError, match="retrieval"):
            builder.build(
                llm=_make_llm(),
                tools=[],
                agent_config=agent_config,
                services={},
            )


class TestKnowledgeGraphMissingKnowledgeConfig:
    """build() raises ValueError when knowledge_config is None or missing default_kb_id."""

    def test_build_raises_when_knowledge_config_is_none(self):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(knowledge_config=None)

        with pytest.raises(ValueError, match="knowledge_config"):
            builder.build(
                llm=_make_llm(),
                tools=[],
                agent_config=agent_config,
                services={"retrieval": _make_retrieval_svc()},
            )

    def test_build_raises_when_default_kb_id_missing(self):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(knowledge_config={"top_k": 8})

        with pytest.raises(ValueError, match="default_kb_id"):
            builder.build(
                llm=_make_llm(),
                tools=[],
                agent_config=agent_config,
                services={"retrieval": _make_retrieval_svc()},
            )

    def test_build_raises_when_knowledge_config_empty_dict(self):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(knowledge_config={})

        with pytest.raises(ValueError, match="default_kb_id"):
            builder.build(
                llm=_make_llm(),
                tools=[],
                agent_config=agent_config,
                services={"retrieval": _make_retrieval_svc()},
            )


class TestKnowledgeGraphBuildSuccess:
    """build() compiles graph when all required config is present."""

    def _build_valid_graph(self, *, tools=None):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(
            knowledge_config={"default_kb_id": _KB_ID, "top_k": 5, "include_parents": True},
            tools=tools,
        )

        return builder.build(
            llm=_make_llm(),
            tools=tools or [],
            agent_config=agent_config,
            services={"retrieval": _make_retrieval_svc()},
        )

    def test_build_returns_compiled_graph(self):
        graph = self._build_valid_graph()
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream_events")

    def test_build_ignores_tools_in_first_phase(self):
        """One-period knowledge graph should compile even when tools are present (just ignored)."""
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def dummy(x: str) -> str:
            """Dummy tool."""
            return x

        graph = self._build_valid_graph(tools=[dummy])
        assert hasattr(graph, "ainvoke")

    def test_graph_nodes_include_retrieve_and_answer(self):
        from baize.agent.graphs.knowledge_react import KnowledgeGraph

        builder = KnowledgeGraph()
        agent_config = _make_agent_config(
            knowledge_config={"default_kb_id": _KB_ID},
        )

        graph = builder.build(
            llm=_make_llm(),
            tools=[],
            agent_config=agent_config,
            services={"retrieval": _make_retrieval_svc()},
        )

        node_names = set(graph.get_graph().nodes.keys())
        assert "retrieve" in node_names
        assert "answer" in node_names


class TestKnowledgeGraphRegistration:
    """@register_graph("knowledge") decorator registers builder correctly."""

    def test_knowledge_agent_type_registered(self):
        import baize.agent.graphs.knowledge_react  # noqa: F401
        from baize.agent.graphs.base import get_graph_builder

        builder = get_graph_builder("knowledge")
        assert builder is not None
        assert builder.agent_type == "knowledge"

    def test_module_imports_without_error(self):
        import importlib
        mod = importlib.import_module("baize.agent.graphs.knowledge_react")
        assert hasattr(mod, "KnowledgeGraph")

    def test_chat_registration_unaffected(self):
        """Registering 'knowledge' should not affect 'chat' registration."""
        import baize.agent.graphs.knowledge_react  # noqa: F401
        from baize.agent.graphs.base import get_graph_builder

        chat_builder = get_graph_builder("chat")
        assert chat_builder.agent_type == "chat"
