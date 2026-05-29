"""Unit tests for build_knowledge_answer_node (F-008)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from baize.agent.graphs.nodes.knowledge_answer import build_knowledge_answer_node
from baize.agent.graphs.state import AgentState


def _make_chunk(*, content: str = "chunk content", idx: int = 0) -> dict:
    return {
        "chunk_id": f"chunk-{idx}",
        "doc_id": f"doc-{idx}",
        "section_path": ["Section A"],
        "score": 0.9,
        "content": content,
    }


def _make_state(messages, *, chunks: list[dict] | None = None) -> AgentState:
    state = AgentState(
        messages=messages,
        user_id="user-1",
        agent_id="agent-1",
    )
    if chunks is not None:
        state["retrieved_chunks"] = chunks
    return state


def _mock_llm(response_content: str = "answer") -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_content))
    return llm


def _get_system_content(messages) -> str | None:
    """Return content of first SystemMessage in message list, or None."""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            return msg.content
    return None


class TestKnowledgeAnswerNodeWithChunks:
    """有 retrieved_chunks 时，system prompt 包含检索资料内容及引用格式要求。"""

    @pytest.mark.asyncio
    async def test_system_prompt_contains_chunk_content(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        chunks = [_make_chunk(content="Python is a programming language", idx=0)]
        state = _make_state([HumanMessage(content="what is python?")], chunks=chunks)
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        system_content = _get_system_content(call_args)
        assert system_content is not None, "Expected a SystemMessage in messages"
        assert "Python is a programming language" in system_content

    @pytest.mark.asyncio
    async def test_system_prompt_contains_citation_format(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        chunks = [_make_chunk(content="some content", idx=0)]
        state = _make_state([HumanMessage(content="question?")], chunks=chunks)
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        system_content = _get_system_content(call_args)
        assert system_content is not None
        assert "[^" in system_content, "System prompt should contain [^数字] citation format instruction"

    @pytest.mark.asyncio
    async def test_multiple_chunks_all_included_in_prompt(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        chunks = [
            _make_chunk(content="First chunk content", idx=0),
            _make_chunk(content="Second chunk content", idx=1),
            _make_chunk(content="Third chunk content", idx=2),
        ]
        state = _make_state([HumanMessage(content="query")], chunks=chunks)
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        system_content = _get_system_content(call_args)
        assert system_content is not None
        assert "First chunk content" in system_content
        assert "Second chunk content" in system_content
        assert "Third chunk content" in system_content

    @pytest.mark.asyncio
    async def test_returns_llm_response_in_messages(self):
        llm = _mock_llm(response_content="Here is the answer [^1].")
        node = build_knowledge_answer_node(llm=llm)

        chunks = [_make_chunk(content="relevant info", idx=0)]
        state = _make_state([HumanMessage(content="query")], chunks=chunks)
        result = await node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Here is the answer [^1]."

    @pytest.mark.asyncio
    async def test_original_messages_preserved_alongside_system(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        human_msg = HumanMessage(content="what is baize?")
        chunks = [_make_chunk(content="Baize is a project", idx=0)]
        state = _make_state([human_msg], chunks=chunks)
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        human_msgs = [m for m in call_args if isinstance(m, HumanMessage)]
        assert any(m.content == "what is baize?" for m in human_msgs)


class TestKnowledgeAnswerNodeWithoutChunks:
    """无 retrieved_chunks 时，system prompt 包含 fallback 文案。"""

    @pytest.mark.asyncio
    async def test_empty_chunks_uses_fallback_prompt(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        state = _make_state([HumanMessage(content="question")], chunks=[])
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        system_content = _get_system_content(call_args)
        assert system_content is not None, "Expected a SystemMessage in messages"
        assert "未检索到相关资料" in system_content

    @pytest.mark.asyncio
    async def test_missing_retrieved_chunks_key_uses_fallback(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        # No retrieved_chunks key in state at all
        state = _make_state([HumanMessage(content="question")], chunks=None)
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        system_content = _get_system_content(call_args)
        assert system_content is not None
        assert "未检索到相关资料" in system_content

    @pytest.mark.asyncio
    async def test_fallback_prompt_mentions_incomplete_info(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        state = _make_state([HumanMessage(content="question")], chunks=[])
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        system_content = _get_system_content(call_args)
        assert system_content is not None
        assert "不完整" in system_content

    @pytest.mark.asyncio
    async def test_no_chunks_still_calls_llm_and_returns_response(self):
        llm = _mock_llm(response_content="I'll answer directly.")
        node = build_knowledge_answer_node(llm=llm)

        state = _make_state([HumanMessage(content="question")], chunks=[])
        result = await node(state)

        llm.ainvoke.assert_awaited_once()
        assert "messages" in result
        assert result["messages"][0].content == "I'll answer directly."


class TestKnowledgeAnswerNodeExistingSystemMessage:
    """已有 SystemMessage 时正确处理（不丢失原有内容）。"""

    @pytest.mark.asyncio
    async def test_existing_system_message_with_chunks(self):
        llm = _mock_llm()
        node = build_knowledge_answer_node(llm=llm)

        chunks = [_make_chunk(content="retrieved info", idx=0)]
        state = _make_state(
            [SystemMessage(content="You are a helpful assistant."), HumanMessage(content="q")],
            chunks=chunks,
        )
        await node(state)

        call_args = llm.ainvoke.call_args[0][0]
        system_content = _get_system_content(call_args)
        assert system_content is not None
        assert "retrieved info" in system_content
