"""Shared LangGraph state types.

All Baize graph variants use this state. Specific graphs may extend it
through ``NotRequired`` fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State for Baize agent graphs.

    Attributes:
        messages: Conversation message history (accumulates via ``add_messages``).
        user_id: The current user's ID.
        agent_id: The agent configuration ID.
        shared_memory: Whether memories saved in this session are shared (visible
                       to all agents for the user) or private to the current agent.
                       Defaults to True when not provided.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    agent_id: str
    shared_memory: NotRequired[bool]
