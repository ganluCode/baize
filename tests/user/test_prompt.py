"""Unit tests for build_user_preferences_prompt."""

import uuid
from unittest.mock import MagicMock

import pytest

from baize.user.prompt import build_user_preferences_prompt


def _make_user(preferences: dict | None, name: str = "ganlu") -> MagicMock:
    user = MagicMock()
    user.name = name
    user.preferences = preferences
    return user


def test_returns_empty_string_when_preferences_none():
    user = _make_user(preferences=None)
    assert build_user_preferences_prompt(user) == ""


def test_contains_name():
    user = _make_user({"communication_style": "formal", "focus_areas": []})
    result = build_user_preferences_prompt(user)
    assert "ganlu" in result


def test_contains_communication_style():
    user = _make_user({"communication_style": "concise", "focus_areas": []})
    result = build_user_preferences_prompt(user)
    assert "concise" in result


def test_contains_focus_areas_when_non_empty():
    user = _make_user({"communication_style": "friendly", "focus_areas": ["AI", "coding"]})
    result = build_user_preferences_prompt(user)
    assert "AI" in result
    assert "coding" in result


def test_no_focus_areas_line_when_empty():
    user = _make_user({"communication_style": "friendly", "focus_areas": []})
    result = build_user_preferences_prompt(user)
    # Should not contain focus_areas/关注领域 line at all
    assert "focus_areas" not in result.lower()
    assert "关注" not in result


def test_contains_custom_instructions_when_set():
    user = _make_user({
        "communication_style": "friendly",
        "focus_areas": [],
        "custom_instructions": "Always reply in English.",
    })
    result = build_user_preferences_prompt(user)
    assert "Always reply in English." in result


def test_no_custom_instructions_line_when_none():
    user = _make_user({"communication_style": "friendly", "focus_areas": [], "custom_instructions": None})
    result = build_user_preferences_prompt(user)
    assert "custom_instructions" not in result.lower()


def test_no_custom_instructions_line_when_empty_string():
    user = _make_user({"communication_style": "friendly", "focus_areas": [], "custom_instructions": ""})
    result = build_user_preferences_prompt(user)
    assert "custom_instructions" not in result.lower()


def test_auto_memory_recall_not_in_output():
    user = _make_user({"communication_style": "friendly", "focus_areas": [], "auto_memory_recall": True})
    result = build_user_preferences_prompt(user)
    assert "auto_memory_recall" not in result


def test_shared_memory_not_in_output():
    user = _make_user({"communication_style": "friendly", "focus_areas": [], "shared_memory": False})
    result = build_user_preferences_prompt(user)
    assert "shared_memory" not in result
