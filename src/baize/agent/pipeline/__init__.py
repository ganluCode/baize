"""Baize agent execution pipeline — sequenced steps around graph execution.

Each step is a :class:`BaseStep` subclass with automatic tracing via
:class:`TraceCollector`. The pipeline runs:

    ContextStep → ReactStep → PersistStep

These are **Baize pipeline steps**, distinct from **LangGraph nodes**
(which live in :mod:`baize.agent.graphs.nodes`).
"""

from baize.agent.pipeline.base import BaseStep, StepContext, StepResult
from baize.agent.pipeline.context import ContextStep
from baize.agent.pipeline.persist import PersistStep
from baize.agent.pipeline.react import ReactStep

__all__ = [
    "BaseStep",
    "ContextStep",
    "PersistStep",
    "ReactStep",
    "StepContext",
    "StepResult",
]
