"""User preferences prompt injection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baize.user.models import UserModel


def build_user_preferences_prompt(user: "UserModel") -> str:
    """Convert user.preferences into a text snippet for injection into Agent System Prompt.

    Args:
        user: UserModel instance with a ``preferences`` dict or None.

    Returns:
        A formatted string describing the user's preferences, or an empty string
        when ``user.preferences`` is None.
    """
    if user.preferences is None:
        return ""

    prefs = user.preferences
    lines: list[str] = []

    lines.append(f"User name: {user.name}")

    communication_style = prefs.get("communication_style")
    if communication_style:
        lines.append(f"Communication style: {communication_style}")

    focus_areas: list[str] = prefs.get("focus_areas") or []
    if focus_areas:
        lines.append(f"Focus areas: {', '.join(focus_areas)}")

    custom_instructions = prefs.get("custom_instructions")
    if custom_instructions:
        lines.append(f"Custom instructions: {custom_instructions}")

    return "\n".join(lines)
