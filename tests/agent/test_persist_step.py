"""Unit tests for PersistStep — message_metadata / citations (F-012)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.agent.pipeline.base import StepContext
from baize.agent.pipeline.persist import PersistStep
from baize.session.models import ChatMessageModel, MessageRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message_model(msg_id: uuid.UUID | None = None) -> ChatMessageModel:
    msg = MagicMock(spec=ChatMessageModel)
    msg.id = msg_id or uuid.uuid4()
    return msg


def _make_ctx(retrieved_chunks: list[dict] | None = None) -> StepContext:
    collector = MagicMock()
    ctx = StepContext(
        collector=collector,
        user_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        message="hello",
    )
    ctx.results["response"] = "assistant response"
    ctx.results["thinking"] = None
    if retrieved_chunks is not None:
        ctx.results["retrieved_chunks"] = retrieved_chunks
    return ctx


def _make_step(session_svc) -> PersistStep:
    return PersistStep(
        session_svc=session_svc,
        session_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
    )


# ---------------------------------------------------------------------------
# message_metadata when no retrieved_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_step_no_chunks_message_metadata_is_none() -> None:
    """When retrieved_chunks is absent, assistant message has message_metadata=None."""
    session_svc = MagicMock()
    session_svc.save_message = AsyncMock(
        side_effect=[_make_message_model(), _make_message_model()]
    )

    ctx = _make_ctx()  # no retrieved_chunks key
    step = _make_step(session_svc)
    await step.run(ctx)

    assistant_kwargs = session_svc.save_message.call_args_list[1].kwargs
    assert assistant_kwargs.get("message_metadata") is None


@pytest.mark.asyncio
async def test_persist_step_empty_chunks_message_metadata_is_none() -> None:
    """When retrieved_chunks is empty list, assistant message has message_metadata=None."""
    session_svc = MagicMock()
    session_svc.save_message = AsyncMock(
        side_effect=[_make_message_model(), _make_message_model()]
    )

    ctx = _make_ctx(retrieved_chunks=[])
    step = _make_step(session_svc)
    await step.run(ctx)

    assistant_kwargs = session_svc.save_message.call_args_list[1].kwargs
    assert assistant_kwargs.get("message_metadata") is None


# ---------------------------------------------------------------------------
# message_metadata when retrieved_chunks is present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_step_chunks_writes_citations() -> None:
    """When retrieved_chunks is non-empty, citations are serialized into message_metadata."""
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "section_path": ["intro", "methods"],
            "score": 0.95,
            "content": "some text",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "section_path": ["results"],
            "score": 0.82,
            "content": "more text",
        },
    ]

    session_svc = MagicMock()
    session_svc.save_message = AsyncMock(
        side_effect=[_make_message_model(), _make_message_model()]
    )

    ctx = _make_ctx(retrieved_chunks=chunks)
    step = _make_step(session_svc)
    await step.run(ctx)

    assistant_kwargs = session_svc.save_message.call_args_list[1].kwargs
    metadata = assistant_kwargs.get("message_metadata")
    assert metadata is not None
    assert "citations" in metadata
    assert len(metadata["citations"]) == 2
    assert metadata["citations"][0] == {
        "chunk_id": "c1",
        "doc_id": "d1",
        "section_path": ["intro", "methods"],
        "score": 0.95,
    }
    assert metadata["citations"][1] == {
        "chunk_id": "c2",
        "doc_id": "d2",
        "section_path": ["results"],
        "score": 0.82,
    }


@pytest.mark.asyncio
async def test_persist_step_citations_exclude_content() -> None:
    """Citations contain only chunk_id/doc_id/section_path/score, not content."""
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "section_path": [],
            "score": 0.7,
            "content": "should not be in metadata",
        }
    ]

    session_svc = MagicMock()
    session_svc.save_message = AsyncMock(
        side_effect=[_make_message_model(), _make_message_model()]
    )

    ctx = _make_ctx(retrieved_chunks=chunks)
    step = _make_step(session_svc)
    await step.run(ctx)

    assistant_kwargs = session_svc.save_message.call_args_list[1].kwargs
    citation = assistant_kwargs["message_metadata"]["citations"][0]
    assert "content" not in citation
    assert set(citation.keys()) == {"chunk_id", "doc_id", "section_path", "score"}


@pytest.mark.asyncio
async def test_persist_step_citations_length_matches_chunks() -> None:
    """message_metadata citations length equals retrieved_chunks length."""
    chunks = [
        {"chunk_id": f"c{i}", "doc_id": f"d{i}", "section_path": [], "score": 0.5, "content": "x"}
        for i in range(5)
    ]

    session_svc = MagicMock()
    session_svc.save_message = AsyncMock(
        side_effect=[_make_message_model(), _make_message_model()]
    )

    ctx = _make_ctx(retrieved_chunks=chunks)
    step = _make_step(session_svc)
    await step.run(ctx)

    assistant_kwargs = session_svc.save_message.call_args_list[1].kwargs
    citations = assistant_kwargs["message_metadata"]["citations"]
    assert len(citations) == 5
