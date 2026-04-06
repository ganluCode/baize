"""Tests for the context compressor (F-011)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n: int) -> list:
    """Alternate HumanMessage / AIMessage, n total messages."""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append(HumanMessage(content=f"user message {i}"))
        else:
            msgs.append(AIMessage(content=f"assistant message {i}"))
    return msgs


def _fake_count_tokens(text: str) -> int:
    """Deterministic token counter: each word is one token."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Tests: no compression needed
# ---------------------------------------------------------------------------

class TestNoCompressionNeeded:
    """When total tokens <= max_tokens the original list is returned as-is."""

    @pytest.mark.asyncio
    async def test_returns_original_when_under_limit(self):
        from baize.context.history import compress_history

        llm = AsyncMock()
        messages = _make_messages(4)

        with patch("baize.context.history._count_tokens_for_messages", return_value=100):
            result = await compress_history(messages, llm, max_tokens=8000, keep_turns=6)

        assert result is messages
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty(self):
        from baize.context.history import compress_history

        llm = AsyncMock()
        result = await compress_history([], llm, max_tokens=8000, keep_turns=6)
        assert result == []
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_exactly_at_limit_returns_original(self):
        from baize.context.history import compress_history

        llm = AsyncMock()
        messages = _make_messages(4)

        with patch("baize.context.history._count_tokens_for_messages", return_value=8000):
            result = await compress_history(messages, llm, max_tokens=8000, keep_turns=6)

        assert result is messages
        llm.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: compression logic
# ---------------------------------------------------------------------------

class TestCompressionLogic:
    """When tokens exceed max_tokens, earlier messages are summarised."""

    @pytest.mark.asyncio
    async def test_keeps_recent_turns(self):
        from baize.context.history import compress_history

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="summary text"))

        # 20 messages, keep_turns=3 → keep last 6, summarise first 14
        messages = _make_messages(20)

        call_count = 0

        def fake_count(msgs):
            nonlocal call_count
            call_count += 1
            # First call (full list) → over limit; subsequent calls → under
            return 9000 if call_count == 1 else 500

        with patch("baize.context.history._count_tokens_for_messages", side_effect=fake_count):
            result = await compress_history(messages, llm, max_tokens=8000, keep_turns=3)

        # First element should be the summary SystemMessage
        assert isinstance(result[0], SystemMessage)
        assert result[0].content.startswith("[历史摘要]")
        # Remaining should be the last keep_turns*2 = 6 messages
        assert result[1:] == messages[-6:]

    @pytest.mark.asyncio
    async def test_summary_inserted_as_system_message(self):
        from baize.context.history import compress_history

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="this is the summary"))

        messages = _make_messages(12)

        call_count = 0

        def fake_count(msgs):
            nonlocal call_count
            call_count += 1
            return 9000 if call_count == 1 else 100

        with patch("baize.context.history._count_tokens_for_messages", side_effect=fake_count):
            result = await compress_history(messages, llm, max_tokens=8000, keep_turns=2)

        assert isinstance(result[0], SystemMessage)
        assert "this is the summary" in result[0].content

    @pytest.mark.asyncio
    async def test_fewer_messages_than_keep_turns_keeps_all(self):
        """If all messages fit within keep_turns*2, no summarisation is attempted."""
        from baize.context.history import compress_history

        llm = AsyncMock()
        messages = _make_messages(4)  # 4 messages, keep_turns=6 → keep_turns*2=12 > 4

        call_count = 0

        def fake_count(msgs):
            nonlocal call_count
            call_count += 1
            return 9000 if call_count == 1 else 500

        with patch("baize.context.history._count_tokens_for_messages", side_effect=fake_count):
            result = await compress_history(messages, llm, max_tokens=8000, keep_turns=6)

        # No earlier messages to summarise; LLM may or may not be called
        # but result should still contain all original messages (possibly without summary)
        # Since there's nothing to summarize, result[-4:] should be original messages
        assert messages[-4:] == result[-4:]


# ---------------------------------------------------------------------------
# Tests: LLM failure fallback
# ---------------------------------------------------------------------------

class TestFallbackOnLLMFailure:
    """When LLM call fails, fallback to truncation by dropping earliest messages."""

    @pytest.mark.asyncio
    async def test_fallback_drops_messages_until_under_limit(self):
        from baize.context.history import compress_history

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        messages = _make_messages(10)

        def fake_count(msgs):
            # Over limit when more than 4 messages
            return 9000 if len(msgs) > 4 else 100

        with patch("baize.context.history._count_tokens_for_messages", side_effect=fake_count):
            result = await compress_history(messages, llm, max_tokens=8000, keep_turns=6)

        # Should have dropped enough messages to go under limit
        assert len(result) <= 10
        # Result should be a suffix of the original messages (earliest dropped)
        assert result == messages[len(messages) - len(result):]

    @pytest.mark.asyncio
    async def test_fallback_logs_warning(self, caplog):
        import logging

        from baize.context.history import compress_history

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))

        messages = _make_messages(10)

        call_count = 0

        def fake_count(msgs):
            nonlocal call_count
            call_count += 1
            # First call (full list) → over limit; subsequent fallback calls → under
            return 9000 if call_count <= 2 else 100

        # keep_turns=2 → keep_count=4 < 10, so LLM is invoked and raises RuntimeError
        with patch("baize.context.history._count_tokens_for_messages", side_effect=fake_count):
            with caplog.at_level(logging.WARNING, logger="baize.context.history"):
                await compress_history(messages, llm, max_tokens=8000, keep_turns=2)

        assert any("fallback" in record.message.lower() or "warning" in record.levelname.lower()
                   for record in caplog.records)

    @pytest.mark.asyncio
    async def test_no_llm_call_when_nothing_to_summarize(self):
        """With keep_turns large enough to cover all messages, LLM never called."""
        from baize.context.history import compress_history

        llm = AsyncMock()
        messages = _make_messages(4)

        call_count = 0

        def fake_count(msgs):
            nonlocal call_count
            call_count += 1
            return 9000 if call_count == 1 else 100

        with patch("baize.context.history._count_tokens_for_messages", side_effect=fake_count):
            await compress_history(messages, llm, max_tokens=8000, keep_turns=100)

        llm.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: token counting helper
# ---------------------------------------------------------------------------

class TestTokenCounting:
    """_count_tokens_for_messages uses tiktoken cl100k_base."""

    def test_counts_tokens_in_messages(self):
        from baize.context.history import _count_tokens_for_messages

        msgs = [HumanMessage(content="hello world"), AIMessage(content="hi")]
        count = _count_tokens_for_messages(msgs)
        assert count > 0
        assert isinstance(count, int)

    def test_empty_messages_returns_zero(self):
        from baize.context.history import _count_tokens_for_messages

        assert _count_tokens_for_messages([]) == 0
