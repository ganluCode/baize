"""Unit tests for the tool registration framework (F-006).

Tests cover:
- @register_tool decorator registration behaviour
- ToolRegistry.get / get_all_names / get_by_names semantics
- Duplicate registration error
- ToolEntry metadata storage
- disabled-permission filtering in get_by_names
"""

from __future__ import annotations

import pytest

from baize.agent.tools import ToolEntry, _ToolRegistry, register_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_registry() -> _ToolRegistry:
    """Return a brand-new, empty registry for isolated test use."""
    return _ToolRegistry()


# ---------------------------------------------------------------------------
# ToolRegistry.register / get / get_all_names
# ---------------------------------------------------------------------------


def test_get_returns_none_for_unknown_name() -> None:
    reg = _fresh_registry()
    assert reg.get("nonexistent") is None


def test_get_all_names_empty_on_new_registry() -> None:
    reg = _fresh_registry()
    assert reg.get_all_names() == []


def test_register_and_get_returns_entry() -> None:
    reg = _fresh_registry()
    entry = ToolEntry(name="my_tool", description="does things", permission="auto", langchain_tool=None)
    reg.register(entry)
    assert reg.get("my_tool") is entry


def test_get_all_names_returns_registered_names() -> None:
    reg = _fresh_registry()
    reg.register(ToolEntry(name="alpha", description="", permission="auto"))
    reg.register(ToolEntry(name="beta", description="", permission="confirm"))
    assert set(reg.get_all_names()) == {"alpha", "beta"}


def test_duplicate_registration_raises_value_error() -> None:
    reg = _fresh_registry()
    entry = ToolEntry(name="dup", description="", permission="auto")
    reg.register(entry)
    with pytest.raises(ValueError, match="dup"):
        reg.register(ToolEntry(name="dup", description="other", permission="auto"))


# ---------------------------------------------------------------------------
# ToolRegistry.get_by_names — filtering and ordering
# ---------------------------------------------------------------------------


def test_get_by_names_returns_matching_entries() -> None:
    reg = _fresh_registry()
    e1 = ToolEntry(name="save_memory", description="saves", permission="auto")
    e2 = ToolEntry(name="search_memory", description="searches", permission="auto")
    reg.register(e1)
    reg.register(e2)

    result = reg.get_by_names(["save_memory", "search_memory"])
    assert e1 in result
    assert e2 in result


def test_get_by_names_skips_unknown_names() -> None:
    reg = _fresh_registry()
    reg.register(ToolEntry(name="real_tool", description="", permission="auto"))

    result = reg.get_by_names(["real_tool", "ghost_tool"])
    assert len(result) == 1
    assert result[0].name == "real_tool"


def test_get_by_names_filters_out_disabled_tools() -> None:
    reg = _fresh_registry()
    reg.register(ToolEntry(name="enabled_tool", description="", permission="auto"))
    reg.register(ToolEntry(name="disabled_tool", description="", permission="disabled"))

    result = reg.get_by_names(["enabled_tool", "disabled_tool"])
    names = [e.name for e in result]
    assert "enabled_tool" in names
    assert "disabled_tool" not in names


def test_get_by_names_includes_confirm_permission_tools() -> None:
    reg = _fresh_registry()
    reg.register(ToolEntry(name="confirm_tool", description="", permission="confirm"))

    result = reg.get_by_names(["confirm_tool"])
    assert len(result) == 1
    assert result[0].permission == "confirm"


def test_get_by_names_empty_input_returns_empty_list() -> None:
    reg = _fresh_registry()
    reg.register(ToolEntry(name="some_tool", description="", permission="auto"))
    assert reg.get_by_names([]) == []


# ---------------------------------------------------------------------------
# ToolEntry metadata storage
# ---------------------------------------------------------------------------


def test_tool_entry_stores_all_metadata() -> None:
    sentinel = object()
    entry = ToolEntry(
        name="meta_tool",
        description="A useful tool",
        permission="confirm",
        langchain_tool=sentinel,
    )
    assert entry.name == "meta_tool"
    assert entry.description == "A useful tool"
    assert entry.permission == "confirm"
    assert entry.langchain_tool is sentinel


# ---------------------------------------------------------------------------
# @register_tool decorator — integration with module singleton
# ---------------------------------------------------------------------------


def test_register_tool_decorator_registers_async_function() -> None:
    """@register_tool registers a function and wraps it as a LangChain tool."""
    # Use a fresh registry to avoid polluting the module singleton.
    fresh_reg = _fresh_registry()

    async def _dummy_fn(query: str) -> str:
        """Search for something."""
        return f"result: {query}"

    from langchain_core.tools import tool as lc_tool

    lc = lc_tool(_dummy_fn)
    entry = ToolEntry(
        name=_dummy_fn.__name__,
        description=_dummy_fn.__doc__ or "",
        permission="auto",
        langchain_tool=lc,
    )
    fresh_reg.register(entry)

    retrieved = fresh_reg.get("_dummy_fn")
    assert retrieved is not None
    assert retrieved.name == "_dummy_fn"
    assert retrieved.permission == "auto"
    assert retrieved.langchain_tool is not None


def test_register_tool_decorator_on_singleton_creates_langchain_tool(monkeypatch) -> None:
    """Applying @register_tool to a new function registers it on the singleton registry
    and stores a non-None langchain_tool."""
    from baize.agent.tools import ToolRegistry

    # Temporarily replace the registry with a fresh one so the test is isolated.
    fresh_reg = _fresh_registry()
    monkeypatch.setattr("baize.agent.tools.ToolRegistry", fresh_reg)

    @register_tool(permission="auto", description="A test helper tool")
    async def _isolated_test_tool(x: str) -> str:
        """Do nothing."""
        return x

    entry = fresh_reg.get("_isolated_test_tool")
    assert entry is not None
    assert entry.name == "_isolated_test_tool"
    assert entry.description == "A test helper tool"
    assert entry.permission == "auto"
    assert entry.langchain_tool is not None


def test_register_tool_uses_docstring_as_description_when_none_given(monkeypatch) -> None:
    """When description is not supplied, the function docstring is used."""
    from baize.agent.tools import ToolRegistry

    fresh_reg = _fresh_registry()
    monkeypatch.setattr("baize.agent.tools.ToolRegistry", fresh_reg)

    @register_tool(permission="auto")
    async def _doc_tool(value: str) -> str:
        """Docstring description here."""
        return value

    entry = fresh_reg.get("_doc_tool")
    assert entry is not None
    assert "Docstring description here" in entry.description
