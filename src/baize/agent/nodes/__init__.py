"""Agent execution nodes with automatic tracing.

All agent pipeline steps inherit from BaseNode, which provides:
- Automatic timing
- Automatic trace reporting via TraceCollector
- Uniform error handling

Business code only implements ``_execute()``, never touches tracing directly.
"""

from baize.agent.nodes.base import BaseNode, NodeContext, NodeResult

__all__ = ["BaseNode", "NodeContext", "NodeResult"]
