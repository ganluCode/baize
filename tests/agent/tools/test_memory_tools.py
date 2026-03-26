"""Unit tests for the built-in memory agent tools (F-008).

Tests cover:
- save_memory returns a confirmation string on success
- save_memory returns an error description when svc.add raises
- save_memory returns a "not initialised" message when service is None
- save_memory returns an error string (not raising) when content is empty
- save_memory passes the correct shared flag from state to svc.add
- search_memory returns a formatted list string on success
- search_memory returns a "not initialised" message when service is None
- search_memory handles empty result list gracefully
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from baize.memory.interface import Memory

_NOW = datetime(2026, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_service(
    add_return: str = "mem-123",
    search_return: list[Memory] | None = None,
    add_side_effect=None,
) -> AsyncMock:
    """Return a mock MemoryServiceInterface."""
    svc = AsyncMock()
    if add_side_effect is not None:
        svc.add.side_effect = add_side_effect
    else:
        svc.add.return_value = add_return
    svc.search.return_value = search_return or []
    return svc


def _state(
    user_id: str = "u-1",
    agent_id: str = "a-1",
    shared_memory: bool = True,
) -> dict:
    """Build a minimal AgentState dict for tool injection."""
    return {"user_id": user_id, "agent_id": agent_id, "shared_memory": shared_memory}


def _config(session_id: str = "s-1") -> dict:
    """Build a minimal RunnableConfig dict for tool injection."""
    return {"configurable": {"thread_id": session_id}}


# ---------------------------------------------------------------------------
# save_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_memory_returns_confirmation_on_success() -> None:
    svc = _make_memory_service(add_return="mem-abc")

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="I love Python", state=_state(), config=_config())

    assert "mem-abc" in result
    assert "I love Python" in result


@pytest.mark.asyncio
async def test_save_memory_returns_error_when_add_raises() -> None:
    svc = _make_memory_service(add_side_effect=RuntimeError("backend error"))

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="some content", state=_state(), config=_config())

    assert "失败" in result or "backend error" in result


@pytest.mark.asyncio
async def test_save_memory_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.memory._get_memory_service", return_value=None):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="something", state=_state(), config=_config())

    assert isinstance(result, str)
    assert len(result) > 0
    assert "初始化" in result or "service" in result.lower()


@pytest.mark.asyncio
async def test_save_memory_empty_content_returns_error_without_calling_service() -> None:
    svc = _make_memory_service()

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="", state=_state(), config=_config())

    assert isinstance(result, str)
    assert len(result) > 0
    svc.add.assert_not_called()


@pytest.mark.asyncio
async def test_save_memory_whitespace_only_content_returns_error() -> None:
    svc = _make_memory_service()

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="   ", state=_state(), config=_config())

    assert isinstance(result, str)
    svc.add.assert_not_called()


@pytest.mark.asyncio
async def test_save_memory_passes_shared_true_when_state_shared_memory_is_true() -> None:
    svc = _make_memory_service(add_return="mem-shared")

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        await save_memory(content="shared info", state=_state(shared_memory=True), config=_config())

    call_kwargs = svc.add.call_args.kwargs
    assert call_kwargs["shared"] is True


@pytest.mark.asyncio
async def test_save_memory_passes_shared_false_when_state_shared_memory_is_false() -> None:
    svc = _make_memory_service(add_return="mem-private")

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        await save_memory(content="private info", state=_state(shared_memory=False), config=_config())

    call_kwargs = svc.add.call_args.kwargs
    assert call_kwargs["shared"] is False


@pytest.mark.asyncio
async def test_save_memory_passes_user_id_agent_id_session_id_to_service() -> None:
    svc = _make_memory_service(add_return="mem-xyz")

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        await save_memory(
            content="test",
            state=_state(user_id="u-99", agent_id="a-77"),
            config=_config(session_id="s-55"),
        )

    call_kwargs = svc.add.call_args.kwargs
    assert call_kwargs["user_id"] == "u-99"
    assert call_kwargs["agent_id"] == "a-77"
    assert call_kwargs["session_id"] == "s-55"


# ---------------------------------------------------------------------------
# search_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_memory_returns_formatted_list_on_success() -> None:
    memories = [
        Memory(
            id="m1", content="I love Python", user_id="u-1",
            score=0.95, metadata={}, created_at=_NOW, updated_at=_NOW,
        ),
        Memory(
            id="m2", content="FastAPI is great", user_id="u-1",
            score=0.82, metadata={}, created_at=_NOW, updated_at=_NOW,
        ),
    ]
    svc = _make_memory_service(search_return=memories)

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import search_memory

        result = await search_memory(query="programming", state=_state(), top_k=5)

    assert "I love Python" in result
    assert "FastAPI is great" in result
    assert "0.95" in result
    assert "0.82" in result


@pytest.mark.asyncio
async def test_search_memory_returns_formatted_list_without_score() -> None:
    """Memories with score=None should omit the score annotation."""
    memories = [
        Memory(
            id="m1", content="No score memory", user_id="u-1",
            score=None, metadata={}, created_at=_NOW, updated_at=_NOW,
        ),
    ]
    svc = _make_memory_service(search_return=memories)

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import search_memory

        result = await search_memory(query="test", state=_state(), top_k=5)

    assert "No score memory" in result
    assert "相关性" not in result


@pytest.mark.asyncio
async def test_search_memory_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.memory._get_memory_service", return_value=None):
        from baize.agent.tools.memory import search_memory

        result = await search_memory(query="something", state=_state(), top_k=5)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_search_memory_handles_empty_results() -> None:
    svc = _make_memory_service(search_return=[])

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import search_memory

        result = await search_memory(query="nothing here", state=_state(), top_k=5)

    assert isinstance(result, str)
    assert "未找到" in result


@pytest.mark.asyncio
async def test_search_memory_passes_user_id_agent_id_to_service() -> None:
    svc = _make_memory_service(search_return=[])

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import search_memory

        await search_memory(
            query="my query",
            state=_state(user_id="u-42", agent_id="a-77"),
            top_k=3,
        )

    svc.search.assert_called_once_with("my query", user_id="u-42", agent_id="a-77", top_k=3)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_memory_tools_are_registered_in_tool_registry() -> None:
    """Both memory tools must be present in the global ToolRegistry."""
    from baize.agent.tools import ToolRegistry
    from baize.agent.tools import memory as _  # noqa: F401 — trigger registration

    names = ToolRegistry.get_all_names()
    assert "save_memory" in names
    assert "search_memory" in names


def test_memory_tools_have_auto_permission() -> None:
    from baize.agent.tools import ToolRegistry
    from baize.agent.tools import memory as _  # noqa: F401

    save_entry = ToolRegistry.get("save_memory")
    search_entry = ToolRegistry.get("search_memory")
    assert save_entry is not None and save_entry.permission == "auto"
    assert search_entry is not None and search_entry.permission == "auto"


def test_save_memory_description_mentions_purpose() -> None:
    """save_memory entry description must clearly convey saving to long-term memory."""
    from baize.agent.tools import ToolRegistry
    from baize.agent.tools import memory as _  # noqa: F401

    entry = ToolRegistry.get("save_memory")
    assert entry is not None
    desc = entry.description
    assert "记忆" in desc or "memory" in desc.lower()
