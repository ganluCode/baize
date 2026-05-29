"""Unit tests for build_knowledge_retrieve_node (F-007)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from baize.agent.graphs.nodes.knowledge_retrieve import build_knowledge_retrieve_node
from baize.agent.graphs.state import AgentState


def _make_scored_chunk(
    *,
    content: str = "chunk content",
    score: float = 0.9,
) -> MagicMock:
    chunk = MagicMock()
    chunk.chunk_id = uuid.uuid4()
    chunk.doc_id = uuid.uuid4()
    chunk.section_path = ["Section A", "Sub B"]
    chunk.score = score
    chunk.content = content
    return chunk


def _make_state(messages) -> AgentState:
    return AgentState(
        messages=messages,
        user_id="user-1",
        agent_id="agent-1",
    )


KB_ID = uuid.uuid4()


class TestKnowledgeRetrieveNodeBasic:
    """正常场景：RetrievalService 返回固定 chunks，state 更新正确。"""

    @pytest.mark.asyncio
    async def test_retrieved_chunks_written_to_state(self):
        chunk = _make_scored_chunk(content="hello knowledge", score=0.85)
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[chunk])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([HumanMessage(content="what is baize?")])
        result = await node(state)

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == 1
        c = result["retrieved_chunks"][0]
        assert c["content"] == "hello knowledge"
        assert c["score"] == pytest.approx(0.85)
        assert c["section_path"] == ["Section A", "Sub B"]
        assert c["chunk_id"] == str(chunk.chunk_id)
        assert c["doc_id"] == str(chunk.doc_id)

    @pytest.mark.asyncio
    async def test_search_called_with_correct_params(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=8,
            include_parents=True,
        )

        state = _make_state([HumanMessage(content="test query")])
        await node(state)

        retrieval_svc.search.assert_awaited_once_with(
            kb_id=KB_ID,
            query="test query",
            top_k=8,
            include_parents=True,
        )

    @pytest.mark.asyncio
    async def test_last_human_message_used_as_query(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([
            HumanMessage(content="first message"),
            AIMessage(content="response"),
            HumanMessage(content="last query"),
        ])
        await node(state)

        call_kwargs = retrieval_svc.search.call_args.kwargs
        assert call_kwargs["query"] == "last query"

    @pytest.mark.asyncio
    async def test_multiple_chunks_all_written(self):
        chunks = [
            _make_scored_chunk(content=f"chunk {i}", score=0.9 - i * 0.1)
            for i in range(3)
        ]
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=chunks)

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([HumanMessage(content="query")])
        result = await node(state)

        assert len(result["retrieved_chunks"]) == 3
        for i, c in enumerate(result["retrieved_chunks"]):
            assert c["content"] == f"chunk {i}"


class TestKnowledgeRetrieveNodeEmptyQuery:
    """query 为空时返回空列表，不调用 search，不抛异常。"""

    @pytest.mark.asyncio
    async def test_no_human_message_returns_empty(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([SystemMessage(content="system"), AIMessage(content="hi")])
        result = await node(state)

        assert result["retrieved_chunks"] == []
        retrieval_svc.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_query_returns_empty(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([HumanMessage(content="")])
        result = await node(state)

        assert result["retrieved_chunks"] == []
        retrieval_svc.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty(self):
        retrieval_svc = AsyncMock()

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([])
        result = await node(state)

        assert result["retrieved_chunks"] == []
        retrieval_svc.search.assert_not_called()


class TestKnowledgeRetrieveNodeFailure:
    """检索失败时记录 warning，返回空列表，不中断流程。"""

    @pytest.mark.asyncio
    async def test_search_exception_returns_empty(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([HumanMessage(content="query")])
        result = await node(state)

        assert result["retrieved_chunks"] == []

    @pytest.mark.asyncio
    async def test_search_exception_logs_warning(self, caplog):
        import logging

        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(side_effect=ConnectionError("Qdrant unreachable"))

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        state = _make_state([HumanMessage(content="query")])

        with caplog.at_level(logging.WARNING, logger="baize.agent.graphs.nodes.knowledge_retrieve"):
            await node(state)

        assert any("warning" in r.levelname.lower() or r.levelno >= logging.WARNING for r in caplog.records)


class TestKnowledgeRetrieveNodeMultiModalContent:
    """content 为 list of blocks 时拍平取文本。"""

    @pytest.mark.asyncio
    async def test_list_content_text_blocks_flattened(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        multimodal_content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        state = _make_state([HumanMessage(content=multimodal_content)])
        await node(state)

        call_kwargs = retrieval_svc.search.call_args.kwargs
        assert "hello" in call_kwargs["query"]
        assert "world" in call_kwargs["query"]

    @pytest.mark.asyncio
    async def test_list_content_with_image_block_extracts_only_text(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        multimodal_content = [
            {"type": "text", "text": "describe this:"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
        state = _make_state([HumanMessage(content=multimodal_content)])
        await node(state)

        call_kwargs = retrieval_svc.search.call_args.kwargs
        assert call_kwargs["query"] == "describe this:"

    @pytest.mark.asyncio
    async def test_list_content_all_non_text_returns_empty(self):
        retrieval_svc = AsyncMock()
        retrieval_svc.search = AsyncMock(return_value=[])

        node = build_knowledge_retrieve_node(
            retrieval_svc=retrieval_svc,
            kb_id=KB_ID,
            top_k=5,
            include_parents=False,
        )

        multimodal_content = [
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
        state = _make_state([HumanMessage(content=multimodal_content)])
        result = await node(state)

        assert result["retrieved_chunks"] == []
        retrieval_svc.search.assert_not_awaited()
