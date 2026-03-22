"""Tool registry for Agent tools.

Provides a singleton ToolRegistry and @register_tool decorator.
The full decorator implementation (F-006) fills in this registry at import time;
this module only exposes the registry interface used by AgentConfigService.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    """Metadata for a registered tool."""

    name: str
    description: str
    permission: str  # "auto" | "confirm" | "disabled"
    langchain_tool: Any = field(default=None)


class _ToolRegistry:
    """Singleton registry of available agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        """Register a tool entry.

        Args:
            entry: The tool entry to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if entry.name in self._tools:
            raise ValueError(f"Tool '{entry.name}' is already registered.")
        self._tools[entry.name] = entry
        logger.debug("Registered tool '%s' (permission=%s).", entry.name, entry.permission)

    def get(self, name: str) -> ToolEntry | None:
        """Return the tool entry for the given name, or None if not found."""
        return self._tools.get(name)

    def get_all_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def get_by_names(self, names: list[str]) -> list[ToolEntry]:
        """Return tool entries for the given names, skipping disabled tools.

        Args:
            names: Tool names to look up.

        Returns:
            List of matching ToolEntry objects, filtering out disabled tools.
        """
        result = []
        for name in names:
            entry = self._tools.get(name)
            if entry is not None and entry.permission != "disabled":
                result.append(entry)
        return result


# Module-level singleton
ToolRegistry = _ToolRegistry()


def register_tool(*, permission: str = "auto", description: str = ""):
    """Decorator that registers an async function as an agent tool.

    Args:
        permission: Permission level — "auto", "confirm", or "disabled".
        description: Human-readable description of the tool.

    Returns:
        Decorator that registers the function and returns it unchanged.
    """

    def decorator(fn):
        from langchain_core.tools import tool as lc_tool

        lc = lc_tool(fn)
        entry = ToolEntry(
            name=fn.__name__,
            description=description or (fn.__doc__ or ""),
            permission=permission,
            langchain_tool=lc,
        )
        ToolRegistry.register(entry)
        return fn

    return decorator
