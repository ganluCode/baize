"""Tests for ContextService.prepare() orchestration."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from baize.context.memory_config import ResolvedMemoryConfig
from baize.context.schemas import PreparedContext
from baize.context.service import ContextService
from baize.memory.config import MemoryConfig
from baize.memory.interface import Memory

_NOW = datetime(2026, 4, 5, 0, 0, 0)


def _make_agent(memory_config=None, tools=None):
    a = MagicMock()
    a.memory_config = memory_config
    a.tools = tools or []
    a.prompts = {"behavior": "Hello {{ user_name }}"}
    return a


def _make_user(name: str = "alice"):
    u = MagicMock()
    u.name = name
    u.preferences = None
    return u


def _make_settings(auto_recall: bool = True, shared: bool = True):
    s = MagicMock()
    s.memory = MemoryConfig(auto_recall=auto_recall, shared=shared)
    return s


def _make_session_svc(history_msgs=None):
    svc = MagicMock()
    svc.get_history = AsyncMock(return_value=history_msgs or [])
    return svc


def _make_memory_svc(memories=None, raise_exc: bool = False):
    svc = MagicMock()
    if raise_exc:
        svc.search = AsyncMock(side_effect=RuntimeError("boom"))
    else:
        svc.search = AsyncMock(return_value=memories or [])
    return svc


def _make_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_prepare_returns_prepared_context():
    agent = _make_agent()
    user = _make_user()
    settings = _make_settings()
    session_svc = _make_session_svc()
    memory_svc = _make_memory_svc()

    svc = ContextService(session_service=session_svc, memory_service=memory_svc, settings=settings)

    result = await svc.prepare(
        agent_config=agent,
        user=user,
        session_id=uuid.uuid4(),
        message="hi",
        llm=_make_llm(),
    )

    assert isinstance(result, PreparedContext)
    assert isinstance(result.system_prompt, str)
    assert result.history == []
    assert result.memories == []
    assert isinstance(result.resolved_memory, ResolvedMemoryConfig)


@pytest.mark.asyncio
async def test_prepare_skips_recall_when_auto_recall_disabled():
    agent = _make_agent(memory_config={"auto_recall": False})
    memory_svc = _make_memory_svc()
    settings = _make_settings(auto_recall=True)

    svc = ContextService(
        session_service=_make_session_svc(),
        memory_service=memory_svc,
        settings=settings,
    )

    await svc.prepare(
        agent_config=agent,
        user=_make_user(),
        session_id=uuid.uuid4(),
        message="hi",
        llm=_make_llm(),
    )

    memory_svc.search.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_recalls_memories_when_enabled():
    memories = [
        Memory(id="1", content="likes python", user_id="u", created_at=_NOW, updated_at=_NOW)
    ]
    agent = _make_agent(memory_config={"auto_recall": True, "top_k": 3})
    memory_svc = _make_memory_svc(memories=memories)

    svc = ContextService(
        session_service=_make_session_svc(),
        memory_service=memory_svc,
        settings=_make_settings(),
    )

    result = await svc.prepare(
        agent_config=agent,
        user=_make_user(),
        session_id=uuid.uuid4(),
        message="query",
        llm=_make_llm(),
    )

    memory_svc.search.assert_awaited_once()
    call_args = memory_svc.search.await_args
    assert call_args.args[0] == "query"
    assert call_args.kwargs.get("top_k") == 3
    assert result.memories == memories


@pytest.mark.asyncio
async def test_prepare_swallows_memory_recall_exception():
    agent = _make_agent(memory_config={"auto_recall": True})
    memory_svc = _make_memory_svc(raise_exc=True)

    svc = ContextService(
        session_service=_make_session_svc(),
        memory_service=memory_svc,
        settings=_make_settings(),
    )

    result = await svc.prepare(
        agent_config=agent,
        user=_make_user(),
        session_id=uuid.uuid4(),
        message="hi",
        llm=_make_llm(),
    )

    assert result.memories == []


@pytest.mark.asyncio
async def test_prepare_works_without_memory_service():
    agent = _make_agent()

    svc = ContextService(
        session_service=_make_session_svc(),
        memory_service=None,
        settings=_make_settings(),
    )

    result = await svc.prepare(
        agent_config=agent,
        user=_make_user(),
        session_id=uuid.uuid4(),
        message="hi",
        llm=_make_llm(),
    )

    assert result.memories == []


@pytest.mark.asyncio
async def test_prepare_calls_compress_history_with_llm():
    agent = _make_agent()
    llm = _make_llm()
    history_msgs = [MagicMock(role="user", content="prior message")]
    # session_service returns ChatMessageModel-like objects; our _history_to_lc_messages
    # only looks at msg.role (enum). Patch compress_history directly.

    svc = ContextService(
        session_service=_make_session_svc(history_msgs=[]),
        memory_service=_make_memory_svc(),
        settings=_make_settings(),
    )

    with patch(
        "baize.context.service.compress_history",
        new=AsyncMock(return_value=[HumanMessage(content="compressed")]),
    ) as compress_mock:
        result = await svc.prepare(
            agent_config=agent,
            user=_make_user(),
            session_id=uuid.uuid4(),
            message="hi",
            llm=llm,
        )

    compress_mock.assert_awaited_once()
    # Second positional arg is llm
    called_with_llm = compress_mock.call_args.args[1]
    assert called_with_llm is llm
    assert result.history[0].content == "compressed"
