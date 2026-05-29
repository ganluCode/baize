"""Unit tests for ReactStep — retrieval_svc injection behavior (F-011)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from baize.agent.models import AgentConfig
from baize.agent.pipeline.base import StepContext
from baize.agent.pipeline.react import ReactStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(agent_type: str = "chat") -> AgentConfig:
    a = AgentConfig()
    a.id = uuid.uuid4()
    a.user_id = uuid.uuid4()
    a.name = "Test Agent"
    a.agent_type = agent_type
    a.tools = {"builtin": [], "mcp_servers": [], "skills": []}
    a.model_config_json = None
    a.knowledge_config = (
        {"default_kb_id": str(uuid.uuid4()), "top_k": 8, "include_parents": True}
        if agent_type == "knowledge"
        else None
    )
    return a


def _make_ctx() -> StepContext:
    from baize.context.memory_config import ResolvedMemoryConfig
    from baize.context.schemas import PreparedContext

    prepared = PreparedContext(
        system_prompt="sys",
        history=[],
        resolved_memory=ResolvedMemoryConfig(auto_memory_recall=False, shared_memory=True),
        memories=[],
    )
    collector = MagicMock()
    collector.add_llm_span = MagicMock()
    collector.finalize = MagicMock()

    ctx = StepContext(
        collector=collector,
        user_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        message="hello",
    )
    ctx.results["prepared"] = prepared
    return ctx


def _minimal_stream():
    """Return an astream_events mock that yields one chat_model_start event."""

    def factory(*args, **kwargs):
        async def gen():
            yield {"event": "on_chat_model_start", "name": "agent", "data": {}}
            yield {
                "event": "on_chat_model_stream",
                "name": "agent",
                "data": {"chunk": AIMessageChunk(content="Hi")},
            }

        return gen()

    return factory


# ---------------------------------------------------------------------------
# Constructor — optional retrieval_svc
# ---------------------------------------------------------------------------


def test_react_step_defaults_retrieval_svc_to_none() -> None:
    """ReactStep can be constructed without retrieval_svc; it defaults to None."""
    step = ReactStep(llm=MagicMock(), agent_config=_make_agent_config())
    assert step._retrieval_svc is None


def test_react_step_stores_retrieval_svc() -> None:
    """ReactStep constructor stores the provided retrieval_svc instance."""
    mock_svc = MagicMock()
    step = ReactStep(
        llm=MagicMock(),
        agent_config=_make_agent_config(),
        retrieval_svc=mock_svc,
    )
    assert step._retrieval_svc is mock_svc


# ---------------------------------------------------------------------------
# builder.build() receives services dict based on retrieval_svc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_step_passes_services_dict_when_retrieval_svc_provided() -> None:
    """builder.build() receives services={"retrieval": svc} when retrieval_svc is set."""
    mock_svc = MagicMock()
    step = ReactStep(
        llm=MagicMock(),
        agent_config=_make_agent_config("knowledge"),
        retrieval_svc=mock_svc,
    )

    mock_graph = MagicMock()
    mock_graph.astream_events = _minimal_stream()
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_graph

    ctx = _make_ctx()
    with (
        patch("baize.agent.pipeline.react.get_graph_builder", return_value=mock_builder),
        patch("baize.agent.pipeline.react.ToolRegistry"),
    ):
        async for _ in step.stream(ctx):
            pass

    services_arg = mock_builder.build.call_args.kwargs.get("services")
    assert services_arg == {"retrieval": mock_svc}


@pytest.mark.asyncio
async def test_react_step_passes_none_services_when_no_retrieval_svc() -> None:
    """builder.build() receives services=None when no retrieval_svc is provided."""
    step = ReactStep(llm=MagicMock(), agent_config=_make_agent_config("chat"))

    mock_graph = MagicMock()
    mock_graph.astream_events = _minimal_stream()
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_graph

    ctx = _make_ctx()
    with (
        patch("baize.agent.pipeline.react.get_graph_builder", return_value=mock_builder),
        patch("baize.agent.pipeline.react.ToolRegistry"),
    ):
        async for _ in step.stream(ctx):
            pass

    services_arg = mock_builder.build.call_args.kwargs.get("services")
    assert services_arg is None
