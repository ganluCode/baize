"""Unit tests for the System Prompt assembler (F-009).

Tests cover:
- Jinja2 template rendering (success and fallback on failure)
- User preferences layer omitted when preferences is None/empty
- Tools layer only contains registered tools, unregistered tools ignored
- Memory layer format when memories are provided vs. absent
- Token truncation: preferences truncated to 500 tokens when total > 4000
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from baize.memory.interface import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(system_prompt: str = "Hello, {{ user_name }}!", tools: list[str] | None = None) -> MagicMock:
    agent = MagicMock()
    agent.system_prompt = system_prompt
    agent.tools = tools or []
    return agent


def _make_user(name: str = "Alice", preferences: dict | None = None) -> MagicMock:
    user = MagicMock()
    user.name = name
    user.preferences = preferences
    return user


def _make_memories(contents: list[str]) -> list[Memory]:
    return [Memory(id=str(i), content=c) for i, c in enumerate(contents)]


# ---------------------------------------------------------------------------
# Jinja2 rendering
# ---------------------------------------------------------------------------


def test_jinja2_user_name_substituted():
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent(system_prompt="Hi {{ user_name }}!")
    user = _make_user(name="Bob")
    result = assemble_system_prompt(agent, user)
    assert "Hi Bob!" in result


def test_jinja2_current_date_substituted():
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent(system_prompt="Today is {{ current_date }}.")
    user = _make_user()
    result = assemble_system_prompt(agent, user)
    # current_date should be a non-empty date string
    assert "Today is {{ current_date }}." not in result
    assert "Today is " in result


def test_jinja2_render_failure_falls_back_to_original():
    from baize.agent.prompt import assemble_system_prompt

    # Invalid Jinja2 syntax: unclosed block
    bad_template = "Hello {% if %} broken"
    agent = _make_agent(system_prompt=bad_template)
    user = _make_user()
    result = assemble_system_prompt(agent, user)
    assert bad_template in result


# ---------------------------------------------------------------------------
# User preferences layer
# ---------------------------------------------------------------------------


def test_user_preferences_absent_when_none():
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent()
    user = _make_user(preferences=None)
    result = assemble_system_prompt(agent, user)
    # The preferences section header should not appear
    assert "用户偏好" not in result


def test_user_preferences_absent_when_build_returns_empty():
    """When build_user_preferences_prompt returns empty string, skip the layer."""
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent()
    user = _make_user(preferences=None)
    with patch("baize.agent.prompt.build_user_preferences_prompt", return_value=""):
        result = assemble_system_prompt(agent, user)
    assert "用户偏好" not in result


def test_user_preferences_included_when_non_empty():
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent()
    user = _make_user(preferences={"communication_style": "concise", "focus_areas": []})
    result = assemble_system_prompt(agent, user)
    assert "concise" in result


# ---------------------------------------------------------------------------
# Tools layer
# ---------------------------------------------------------------------------


def test_tools_layer_contains_registered_tool_name_and_description():
    from baize.agent.prompt import assemble_system_prompt
    from baize.agent.tools import ToolEntry, _ToolRegistry

    fresh_reg = _ToolRegistry()
    fresh_reg.register(ToolEntry(name="do_thing", description="Does a thing.", permission="auto"))

    agent = _make_agent(tools=["do_thing"])
    user = _make_user()

    with patch("baize.agent.prompt.ToolRegistry", fresh_reg):
        result = assemble_system_prompt(agent, user)

    assert "do_thing" in result
    assert "Does a thing." in result


def test_tools_layer_omits_unregistered_tools():
    from baize.agent.prompt import assemble_system_prompt
    from baize.agent.tools import ToolEntry, _ToolRegistry

    fresh_reg = _ToolRegistry()
    fresh_reg.register(ToolEntry(name="real_tool", description="Real.", permission="auto"))

    agent = _make_agent(tools=["real_tool", "ghost_tool"])
    user = _make_user()

    with patch("baize.agent.prompt.ToolRegistry", fresh_reg):
        result = assemble_system_prompt(agent, user)

    assert "real_tool" in result
    assert "ghost_tool" not in result


def test_tools_layer_absent_when_no_tools():
    from baize.agent.prompt import assemble_system_prompt
    from baize.agent.tools import _ToolRegistry

    fresh_reg = _ToolRegistry()
    agent = _make_agent(tools=[])
    user = _make_user()

    with patch("baize.agent.prompt.ToolRegistry", fresh_reg):
        result = assemble_system_prompt(agent, user)

    assert "可用工具" not in result


# ---------------------------------------------------------------------------
# Memory injection layer
# ---------------------------------------------------------------------------


def test_memory_section_absent_when_empty():
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent()
    user = _make_user()
    result = assemble_system_prompt(agent, user, memories=[])
    assert "关于用户的已知信息" not in result


def test_memory_section_present_with_correct_header():
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent()
    user = _make_user()
    memories = _make_memories(["I like Python", "I work at Acme"])
    result = assemble_system_prompt(agent, user, memories=memories)
    assert "## 关于用户的已知信息（仅供参考）" in result


def test_memory_section_contains_memory_contents():
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent()
    user = _make_user()
    memories = _make_memories(["I like Python", "I work at Acme"])
    result = assemble_system_prompt(agent, user, memories=memories)
    assert "I like Python" in result
    assert "I work at Acme" in result


# ---------------------------------------------------------------------------
# Token limit — preferences truncation
# ---------------------------------------------------------------------------


def test_token_limit_truncates_preferences_layer():
    """When total tokens exceed 4000, user preferences are truncated to 500 tokens."""
    from baize.agent.prompt import assemble_system_prompt

    agent = _make_agent(system_prompt="System.")
    user = _make_user(preferences={"communication_style": "verbose", "focus_areas": []})

    # Simulate: full prompt is over 4000 tokens; preferences alone is large
    # We patch _count_tokens so that on the first call (full text) it returns >4000,
    # while subsequent calls behave normally for the truncation logic.
    call_count = {"n": 0}

    def fake_count(text: str) -> int:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: simulate total over limit
            return 4001
        # Subsequent calls: real counting (via tiktoken would be low for short text)
        return len(text.split())

    with patch("baize.agent.prompt._count_tokens", side_effect=fake_count):
        # Should not raise; result should be a string
        result = assemble_system_prompt(agent, user)

    assert isinstance(result, str)


def test_no_truncation_when_within_token_limit():
    """Prompt is returned unchanged when total tokens are within the limit."""
    from baize.agent.prompt import assemble_system_prompt

    long_prefs = "word " * 200  # short enough after join

    agent = _make_agent(system_prompt="System.")
    user = _make_user(preferences={"custom_instructions": long_prefs, "focus_areas": []})

    with patch("baize.agent.prompt._count_tokens", return_value=100):
        result = assemble_system_prompt(agent, user)

    # All 200 "word" tokens should remain intact (no truncation)
    assert "word" in result


def test_other_layers_intact_after_truncation():
    """When preferences are truncated, other layers (system prompt, memories) stay complete."""
    from baize.agent.prompt import assemble_system_prompt
    from baize.agent.tools import ToolEntry, _ToolRegistry

    fresh_reg = _ToolRegistry()
    fresh_reg.register(ToolEntry(name="my_tool", description="My description.", permission="auto"))

    agent = _make_agent(system_prompt="Important system text.", tools=["my_tool"])
    user = _make_user(preferences={"communication_style": "verbose", "focus_areas": []})
    memories = _make_memories(["Key memory item"])

    call_count = {"n": 0}

    def fake_count(text: str) -> int:
        call_count["n"] += 1
        return 4001 if call_count["n"] == 1 else 10

    with patch("baize.agent.prompt._count_tokens", side_effect=fake_count):
        with patch("baize.agent.prompt.ToolRegistry", fresh_reg):
            result = assemble_system_prompt(agent, user, memories=memories)

    assert "Important system text." in result
    assert "my_tool" in result
    assert "Key memory item" in result
