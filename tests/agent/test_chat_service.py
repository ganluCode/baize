"""Unit tests for AgentService.chat() orchestration logic."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessageChunk

from baize.agent.models import AgentConfig
from baize.agent.service import AgentConfigService, AgentService, AgentServiceError, ChatEvent
from baize.memory.interface import Memory, MemoryServiceInterface
from baize.session.models import MessageRole, SessionModel
from baize.session.service import SessionService
from baize.user.models import UserModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    user_id: uuid.UUID | None = None,
    agent_id: uuid.UUID | None = None,
    auto_memory_recall: bool = False,
    tools: list[str] | None = None,
) -> AgentConfig:
    a = AgentConfig()
    a.id = agent_id or uuid.uuid4()
    a.user_id = user_id or uuid.uuid4()
    a.name = "Test Agent"
    a.description = None
    a.agent_type = "chat"
    a.is_enabled = True
    a.prompts = {"behavior": "You are helpful."}
    a.tools = {"builtin": tools or [], "mcp_servers": [], "skills": []}
    a.model_config_json = None
    a.sub_agents = None
    a.memory_config = {"auto_recall": auto_memory_recall, "shared": True, "top_k": 5}
    a.guardrails = {"max_tool_calls": 10, "timeout_seconds": 120}
    a.created_at = datetime.now(UTC)
    a.updated_at = datetime.now(UTC)
    return a


def _make_session(user_id: uuid.UUID, agent_id: uuid.UUID) -> SessionModel:
    s = SessionModel()
    s.id = uuid.uuid4()
    s.user_id = user_id
    s.agent_id = agent_id
    s.title = None
    s.title_gen_attempts = 0
    from baize.session.models import SessionStatus

    s.status = SessionStatus.active
    s.created_at = datetime.now(UTC)
    s.updated_at = datetime.now(UTC)
    return s


def _make_user(user_id: uuid.UUID | None = None) -> UserModel:
    u = UserModel()
    u.id = user_id or uuid.uuid4()
    u.email = "test@example.com"
    u.name = "TestUser"
    u.password = "hashed"
    u.role = "user"
    u.avatar = None
    u.api_key_hash = None
    u.preferences = None
    u.is_active = True
    u.created_at = datetime.now(UTC)
    u.updated_at = datetime.now(UTC)
    return u


def _make_service(
    agent_config_svc: AgentConfigService | None = None,
    session_svc: SessionService | None = None,
    memory_svc: MemoryServiceInterface | None = None,
    model_router=None,
    context_svc=None,
) -> AgentService:
    if model_router is None:
        mr = MagicMock()
        mr.get_chat_model.return_value = MagicMock()
        model_router = mr
    if context_svc is None:
        from baize.context.memory_config import ResolvedMemoryConfig
        from baize.context.schemas import PreparedContext

        ctx = MagicMock()
        ctx.prepare = AsyncMock(
            return_value=PreparedContext(
                system_prompt="sys",
                history=[],
                resolved_memory=ResolvedMemoryConfig(
                    auto_memory_recall=True, shared_memory=True
                ),
                memories=[],
            )
        )
        context_svc = ctx
    return AgentService(
        agent_config_service=agent_config_svc or AsyncMock(spec=AgentConfigService),
        session_service=session_svc or AsyncMock(spec=SessionService),
        memory_service=memory_svc,
        model_router=model_router,
        context_service=context_svc,
    )


def _make_simple_stream(*event_dicts):
    """Return an async generator factory that yields the given event dicts."""

    def factory(*args, **kwargs):
        async def gen():
            for e in event_dicts:
                yield e

        return gen()

    return factory


async def _collect(gen) -> list[ChatEvent]:
    """Collect all ChatEvents from an async generator."""
    events = []
    async for event in gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# ChatEvent dataclass
# ---------------------------------------------------------------------------


def test_chat_event_has_type_and_payload() -> None:
    event = ChatEvent(type="token", payload={"content": "hello"})
    assert event.type == "token"
    assert event.payload == {"content": "hello"}


# ---------------------------------------------------------------------------
# agent_id not found / not owned → error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_yields_error_on_unknown_agent() -> None:
    """chat() yields an error event when agent_id not found or not owned."""
    user_id = uuid.uuid4()
    user = _make_user(user_id)

    agent_config_svc = AsyncMock(spec=AgentConfigService)
    agent_config_svc.get.side_effect = AgentServiceError("Agent not found.", status_code=404)

    svc = _make_service(agent_config_svc=agent_config_svc)

    events = await _collect(svc.chat(uuid.uuid4(), uuid.uuid4(), user_id, "hello", user))

    assert len(events) == 1
    assert events[0].type == "error"
    agent_config_svc.get.assert_called_once()


# ---------------------------------------------------------------------------
# session not found → auto-create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_auto_creates_session_when_not_found() -> None:
    """chat() creates a new session when get_session raises 404."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    user = _make_user(user_id)
    agent = _make_agent(user_id=user_id, agent_id=agent_id)
    session = _make_session(user_id, agent_id)

    agent_config_svc = AsyncMock(spec=AgentConfigService)
    agent_config_svc.get.return_value = agent

    session_svc = AsyncMock(spec=SessionService)
    session_svc.get_session.side_effect = HTTPException(status_code=404, detail="Session not found.")
    session_svc.create_session.return_value = session
    session_svc.get_history.return_value = []
    session_svc.save_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))

    mock_graph = MagicMock()
    mock_graph.astream_events = _make_simple_stream(
        {"event": "on_chat_model_start", "name": "agent", "data": {}},
        {"event": "on_chat_model_stream", "name": "agent", "data": {"chunk": AIMessageChunk(content="Hi")}},
    )

    model_router = MagicMock()
    model_router.get_chat_model.return_value = MagicMock()
    svc = _make_service(agent_config_svc=agent_config_svc, session_svc=session_svc, model_router=model_router)

    with (
        patch("baize.agent.nodes.react_node.build_react_graph", return_value=mock_graph),
        patch("baize.agent.nodes.react_node.ToolRegistry"),
    ):
        await _collect(svc.chat(agent_id, uuid.uuid4(), user_id, "hello", user))

    session_svc.create_session.assert_called_once_with(user_id, agent_id)


# Memory recall is now handled by ContextService; see tests/context/test_service.py
# for tests covering auto_memory_recall behavior.


# ---------------------------------------------------------------------------
# tool call limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_yields_error_when_tool_call_limit_exceeded() -> None:
    """chat() yields an error event after 10 tool calls and stops."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    user = _make_user(user_id)
    agent = _make_agent(user_id=user_id, agent_id=agent_id)
    session = _make_session(user_id, agent_id)

    agent_config_svc = AsyncMock(spec=AgentConfigService)
    agent_config_svc.get.return_value = agent

    session_svc = AsyncMock(spec=SessionService)
    session_svc.get_session.return_value = session
    session_svc.get_history.return_value = []
    session_svc.save_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))

    # Produce 11 tool_start events, exceeding the 10-call limit
    tool_events = [
        {"event": "on_tool_start", "name": f"tool_{i}", "data": {"input": {}}}
        for i in range(11)
    ]
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_simple_stream(*tool_events)

    model_router = MagicMock()
    model_router.get_chat_model.return_value = MagicMock()
    svc = _make_service(agent_config_svc=agent_config_svc, session_svc=session_svc, model_router=model_router)

    with (
        patch("baize.agent.nodes.react_node.build_react_graph", return_value=mock_graph),
        patch("baize.agent.nodes.react_node.ToolRegistry"),
    ):
        events = await _collect(svc.chat(agent_id, uuid.uuid4(), user_id, "hello", user))

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1, f"Expected exactly 1 error event, got: {events}"
    # Streaming should stop after the error — no "done" event
    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 0


# ---------------------------------------------------------------------------
# messages saved after successful conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_saves_user_and_assistant_messages_after_completion() -> None:
    """chat() persists user message and assistant reply to the session at the end."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    user = _make_user(user_id)
    agent = _make_agent(user_id=user_id, agent_id=agent_id)
    session = _make_session(user_id, agent_id)
    saved_msg = MagicMock(id=uuid.uuid4())

    agent_config_svc = AsyncMock(spec=AgentConfigService)
    agent_config_svc.get.return_value = agent

    session_svc = AsyncMock(spec=SessionService)
    session_svc.get_session.return_value = session
    session_svc.get_history.return_value = []
    session_svc.save_message = AsyncMock(return_value=saved_msg)

    mock_graph = MagicMock()
    mock_graph.astream_events = _make_simple_stream(
        {"event": "on_chat_model_start", "name": "agent", "data": {}},
        {"event": "on_chat_model_stream", "name": "agent", "data": {"chunk": AIMessageChunk(content="Hello!")}},
    )

    model_router = MagicMock()
    model_router.get_chat_model.return_value = MagicMock()
    svc = _make_service(agent_config_svc=agent_config_svc, session_svc=session_svc, model_router=model_router)

    with (
        patch("baize.agent.nodes.react_node.build_react_graph", return_value=mock_graph),
        patch("baize.agent.nodes.react_node.ToolRegistry"),
    ):
        events = await _collect(svc.chat(agent_id, uuid.uuid4(), user_id, "hello", user))

    # Two save_message calls: one for user, one for assistant
    assert session_svc.save_message.call_count == 2

    call_roles = [call.kwargs["role"] for call in session_svc.save_message.call_args_list]
    assert MessageRole.user in call_roles
    assert MessageRole.assistant in call_roles

    # done event should contain session_id and message_id
    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
    assert "session_id" in done_events[0].payload
    assert "message_id" in done_events[0].payload


# ---------------------------------------------------------------------------
# timeout → error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_yields_error_on_timeout() -> None:
    """chat() yields an error event when the conversation times out."""
    user_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    user = _make_user(user_id)
    agent = _make_agent(user_id=user_id, agent_id=agent_id)
    session = _make_session(user_id, agent_id)

    agent_config_svc = AsyncMock(spec=AgentConfigService)
    agent_config_svc.get.return_value = agent

    session_svc = AsyncMock(spec=SessionService)
    session_svc.get_session.return_value = session
    session_svc.get_history.return_value = []

    # Graph that raises TimeoutError (simulating asyncio timeout)
    def timeout_stream(*args, **kwargs):
        async def gen():
            raise TimeoutError("timed out")
            yield  # make it a generator

        return gen()

    mock_graph = MagicMock()
    mock_graph.astream_events = timeout_stream

    model_router = MagicMock()
    model_router.get_chat_model.return_value = MagicMock()
    svc = _make_service(agent_config_svc=agent_config_svc, session_svc=session_svc, model_router=model_router)

    with (
        patch("baize.agent.nodes.react_node.build_react_graph", return_value=mock_graph),
        patch("baize.agent.nodes.react_node.ToolRegistry"),
    ):
        events = await _collect(svc.chat(agent_id, uuid.uuid4(), user_id, "hello", user))

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
