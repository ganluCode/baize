"""Export Mermaid diagrams for all registered agent graphs.

Usage:
    uv run python scripts/export_graphs.py

Writes ``docs/graphs/<agent_type>.md`` for each registered agent_type.
Each file contains:
  - Mermaid diagram (renderable on GitHub / IDE preview)
  - Builder class info
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from baize.agent.graphs import get_graph_builder, list_agent_types

OUT_DIR = Path(__file__).parents[1] / "docs" / "graphs"


def _make_mock_llm():
    """Create a minimal LLM mock — bind_tools returns itself so build() doesn't fail."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.__class__.__name__ = "ChatAnthropic"
    return llm


def _make_mock_agent_config():
    cfg = MagicMock()
    cfg.tools = None
    cfg.agent_type = "chat"
    return cfg


def export_graph(agent_type: str) -> str:
    """Build the graph and return its Mermaid source."""
    builder = get_graph_builder(agent_type)
    graph = builder.build(
        llm=_make_mock_llm(),
        tools=[],
        agent_config=_make_mock_agent_config(),
    )
    return graph.get_graph().draw_mermaid()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agent_types = list_agent_types()
    print(f"Found {len(agent_types)} agent types: {agent_types}")

    index_lines = ["# Agent Graphs\n", "Auto-generated from registered `GraphBuilder` subclasses.\n\n"]

    for agent_type in agent_types:
        builder = get_graph_builder(agent_type)
        builder_class = builder.__class__.__name__
        mermaid = export_graph(agent_type)

        out_file = OUT_DIR / f"{agent_type}.md"
        out_file.write_text(
            f"# `{agent_type}` Agent Graph\n\n"
            f"**Builder**: `{builder_class}` "
            f"(`baize.agent.graphs.{builder.__class__.__module__.rsplit('.', 1)[-1]}`)\n\n"
            f"```mermaid\n{mermaid}\n```\n",
            encoding="utf-8",
        )
        print(f"  ✓ {out_file.relative_to(Path.cwd())}")
        index_lines.append(f"- [`{agent_type}`]({agent_type}.md) — {builder_class}\n")

    (OUT_DIR / "README.md").write_text("".join(index_lines), encoding="utf-8")
    print(f"  ✓ {(OUT_DIR / 'README.md').relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
