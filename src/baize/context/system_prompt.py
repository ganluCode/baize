"""System Prompt layered assembler.

Layers (in order):
  1a. Soul — identity / persona layer from ``agent_config.prompts.soul``.
  1b. Behavior — task / behavior instructions from ``agent_config.prompts.behavior``.
      Both layers are Jinja2 templates rendered with (user_name, current_date).
      Either layer may be absent.
  2. User preferences — formatted snippet from :func:`build_user_preferences_prompt`.
  3. Tool descriptions — name + description for each registered tool listed in
     ``agent_config.tools``; unregistered tools are silently ignored.
  4. Memory injection — relevant memories retrieved before the chat round.

Token budget: when the assembled prompt exceeds *TOKEN_LIMIT* tokens the user
preferences layer is truncated to at most *PREFS_TOKEN_LIMIT* tokens; all other
layers are left intact.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

from baize.agent.tools import ToolRegistry
from baize.context.user_prefs import build_user_preferences_prompt

if TYPE_CHECKING:
    from baize.agent.models import AgentConfig
    from baize.memory.interface import Memory
    from baize.user.models import UserModel

logger = logging.getLogger(__name__)

TOKEN_LIMIT: int = 4000
PREFS_TOKEN_LIMIT: int = 500


def _count_tokens(text: str) -> int:
    """Count tokens in *text* using tiktoken cl100k_base encoding."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return *text* truncated so it contains at most *max_tokens* tokens."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _render_system_prompt(raw: str, user_name: str) -> str:
    """Render *raw* as a Jinja2 template, falling back to the original on error."""
    try:
        from jinja2 import Template

        tmpl = Template(raw)
        return tmpl.render(user_name=user_name, current_date=date.today().isoformat())
    except Exception:  # noqa: BLE001
        logger.warning("Jinja2 rendering failed; using raw system_prompt.")
        return raw


def assemble_system_prompt(
    agent_config: AgentConfig,
    user: UserModel,
    memories: list[Memory] | None = None,
) -> str:
    """Assemble the final system prompt from multiple layers.

    Args:
        agent_config: The loaded AgentConfig ORM object.
        user: The current UserModel.
        memories: Optional list of Memory objects to inject.

    Returns:
        The fully assembled system prompt string.
    """
    if memories is None:
        memories = []

    # ------------------------------------------------------------------
    # Layer 1a + 1b: Soul and Behavior (both optional, from prompts dict)
    # ------------------------------------------------------------------
    prompts = agent_config.prompts or {}
    soul_raw = prompts.get("soul") if isinstance(prompts, dict) else None
    behavior_raw = prompts.get("behavior") if isinstance(prompts, dict) else None

    layer_soul = _render_system_prompt(soul_raw, user.name) if soul_raw else ""
    layer_behavior = _render_system_prompt(behavior_raw, user.name) if behavior_raw else ""

    # ------------------------------------------------------------------
    # Layer 2: User preferences
    # ------------------------------------------------------------------
    layer_prefs = build_user_preferences_prompt(user)
    if layer_prefs:
        layer_prefs_section = f"## 用户偏好\n{layer_prefs}"
    else:
        layer_prefs_section = ""

    # ------------------------------------------------------------------
    # Layer 3: Tool descriptions
    # ------------------------------------------------------------------
    tool_names: list[str] = list(agent_config.tools or [])
    tool_entries = ToolRegistry.get_by_names(tool_names)
    if tool_entries:
        tool_lines = [f"- {e.name}: {e.description}" for e in tool_entries]
        layer_tools = "## 可用工具\n" + "\n".join(tool_lines)
    else:
        layer_tools = ""

    # ------------------------------------------------------------------
    # Layer 4: Memory injection
    # ------------------------------------------------------------------
    if memories:
        mem_lines = [f"{i}. {m.content}" for i, m in enumerate(memories, start=1)]
        layer_memories = "## 关于用户的已知信息（仅供参考）\n" + "\n".join(mem_lines)
    else:
        layer_memories = ""

    # ------------------------------------------------------------------
    # Token budget: truncate preferences if total exceeds limit
    # ------------------------------------------------------------------
    parts = [layer_soul, layer_behavior, layer_prefs_section, layer_tools, layer_memories]
    full_text = "\n\n".join(p for p in parts if p)

    if _count_tokens(full_text) > TOKEN_LIMIT and layer_prefs_section:
        truncated_prefs_text = _truncate_to_tokens(layer_prefs_section, PREFS_TOKEN_LIMIT)
        parts = [layer_soul, layer_behavior, truncated_prefs_text, layer_tools, layer_memories]
        full_text = "\n\n".join(p for p in parts if p)

    return full_text
