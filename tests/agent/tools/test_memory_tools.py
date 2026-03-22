"""Unit tests for the built-in memory agent tools (F-007).

Tests cover:
- save_memory returns a confirmation string on success
- save_memory returns an error description when add_memory raises
- save_memory returns a "not initialised" message when service is None
- search_memory returns a formatted list string on success
- search_memory returns a "not initialised" message when service is None
- search_memory handles empty result list gracefully
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from baize.memory.interface import Memory


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
        svc.add_memory.side_effect = add_side_effect
    else:
        svc.add_memory.return_value = add_return
    svc.search.return_value = search_return or []
    return svc


# ---------------------------------------------------------------------------
# save_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_memory_returns_confirmation_on_success() -> None:
    svc = _make_memory_service(add_return="mem-abc")

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="I love Python", metadata={})

    assert "mem-abc" in result or "saved" in result.lower() or "记忆" in result


@pytest.mark.asyncio
async def test_save_memory_returns_error_when_add_memory_raises() -> None:
    svc = _make_memory_service(add_side_effect=RuntimeError("backend error"))

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="some content", metadata={})

    assert "error" in result.lower() or "失败" in result or "backend error" in result


@pytest.mark.asyncio
async def test_save_memory_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.memory._get_memory_service", return_value=None):
        from baize.agent.tools.memory import save_memory

        result = await save_memory(content="something", metadata={})

    # Should not crash; should return a user-friendly message
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_save_memory_passes_metadata_to_add_memory() -> None:
    svc = _make_memory_service()
    meta = {"user_id": "u-1", "source": "chat"}

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import save_memory

        await save_memory(content="test content", metadata=meta)

    svc.add_memory.assert_called_once_with("test content", meta)


# ---------------------------------------------------------------------------
# search_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_memory_returns_formatted_list_on_success() -> None:
    memories = [
        Memory(id="m1", content="I love Python", metadata={}),
        Memory(id="m2", content="FastAPI is great", metadata={}),
    ]
    svc = _make_memory_service(search_return=memories)

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import search_memory

        result = await search_memory(query="programming", top_k=5)

    assert "I love Python" in result
    assert "FastAPI is great" in result


@pytest.mark.asyncio
async def test_search_memory_returns_not_initialised_when_service_is_none() -> None:
    with patch("baize.agent.tools.memory._get_memory_service", return_value=None):
        from baize.agent.tools.memory import search_memory

        result = await search_memory(query="something", top_k=5)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_search_memory_handles_empty_results() -> None:
    svc = _make_memory_service(search_return=[])

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import search_memory

        result = await search_memory(query="nothing here", top_k=5)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_search_memory_passes_query_and_top_k_to_service() -> None:
    svc = _make_memory_service(search_return=[])

    with patch("baize.agent.tools.memory._get_memory_service", return_value=svc):
        from baize.agent.tools.memory import search_memory

        await search_memory(query="my query", top_k=3)

    svc.search.assert_called_once_with("my query", top_k=3)


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
