"""Dynamic option resolvers for metadata fields.

ResolverContext is constructed per-request and provides methods to fetch
options from code, config files, or the database.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from baize.metadata.schemas import FieldOption

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from baize.core.config import Settings
    from baize.llm.provider import ProviderFactory


@dataclass
class ResolverContext:
    """Per-request context for resolving dynamic field options."""

    user_id: uuid.UUID
    db: "AsyncSession"
    provider_factory: "ProviderFactory"
    settings: "Settings"

    # ------------------------------------------------------------------
    # Source A: code-defined (ToolRegistry)
    # ------------------------------------------------------------------

    def get_builtin_tool_options(self) -> list[FieldOption]:
        from baize.agent.tools import ToolRegistry

        entries = ToolRegistry.get_by_names(ToolRegistry.get_all_names())
        return [
            FieldOption(value=e.name, label=f"{e.name} — {e.description}" if e.description else e.name)
            for e in entries
        ]

    # ------------------------------------------------------------------
    # Source B: config-file driven (ProviderFactory, yaml_config)
    # ------------------------------------------------------------------

    def get_model_options(self, usage: str = "chat") -> list[FieldOption]:
        options = []
        for p in self.provider_factory.list_providers():
            if p["status"] == "unavailable":
                continue
            for m in p["models"]:
                if usage in m["usage"]:
                    value = f"{p['name']}/{m['id']}"
                    label = f"{p['name']} / {m['id']}"
                    options.append(FieldOption(value=value, label=label))
        return options

    def get_mcp_server_options(self) -> list[FieldOption]:
        mcp_cfg = self.settings.yaml_config.get("mcp_servers", {})
        if isinstance(mcp_cfg, dict):
            return [FieldOption(value=name, label=name) for name in mcp_cfg]
        if isinstance(mcp_cfg, list):
            return [FieldOption(value=s.get("name", str(s)), label=s.get("name", str(s))) for s in mcp_cfg]
        return []

    # ------------------------------------------------------------------
    # Source C: database-driven
    # ------------------------------------------------------------------

    async def get_user_agent_options(self) -> list[FieldOption]:
        from baize.agent.models import AgentConfig
        from sqlalchemy import select

        result = await self.db.execute(
            select(AgentConfig.id, AgentConfig.name)
            .where(AgentConfig.user_id == self.user_id)
            .order_by(AgentConfig.name)
        )
        return [
            FieldOption(value=str(row.id), label=row.name)
            for row in result.all()
        ]
