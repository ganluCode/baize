"""Context compressor for Agent conversation history.

compress_history() trims a message list to stay within a token budget:
  1. If total tokens <= max_tokens, return the original list unchanged.
  2. Otherwise keep the most recent keep_turns*2 messages intact and summarise
     all earlier messages using the LLM (inserted as a SystemMessage).
  3. If the LLM call fails, fall back to dropping the earliest messages one by
     one until the total drops below max_tokens, logging a warning.

Token counting uses tiktoken cl100k_base (model-agnostic).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

_SUMMARISE_PROMPT = (
    "请用简洁的中文总结以下对话历史的关键信息，供AI助手参考。"
    "只保留重要事实、用户偏好和上下文，不需要逐条复述。\n\n"
    "{history}"
)


def _count_tokens_for_messages(messages: list[BaseMessage]) -> int:
    """Count total tokens across all message contents using cl100k_base."""
    if not messages:
        return 0
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(enc.encode(content))
    return total


async def compress_history(
    messages: list[BaseMessage],
    llm: BaseLanguageModel,
    max_tokens: int = 8000,
    keep_turns: int = 6,
) -> list[BaseMessage]:
    """Compress conversation history to stay within *max_tokens*.

    Args:
        messages: Full conversation history (list of BaseMessage).
        llm: LangChain LLM used to generate a summary of earlier messages.
        max_tokens: Maximum total token budget for the returned list.
        keep_turns: Number of recent conversation turns to preserve intact.
            Each turn consists of one user message + one assistant message,
            so keep_turns=6 preserves at most 12 messages.

    Returns:
        A (possibly compressed) list of BaseMessage objects.  The original
        list is returned unchanged when it already fits within *max_tokens*.
    """
    if not messages:
        return messages

    if _count_tokens_for_messages(messages) <= max_tokens:
        return messages

    # Number of messages to keep at the end (each "turn" = 2 messages)
    keep_count = keep_turns * 2
    if keep_count >= len(messages):
        # Nothing to summarise; fall straight to truncation fallback
        return await _fallback_truncate(messages, max_tokens)

    recent = messages[-keep_count:]
    earlier = messages[:-keep_count]

    # --- Attempt LLM summarisation ---
    try:
        # Build a plain history string (role: content)
        lines = []
        for m in earlier:
            role = "助手" if isinstance(m, AIMessage) else "用户"
            content = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

        prompt = _SUMMARISE_PROMPT.format(history=history_text)
        response = await llm.ainvoke(prompt)
        summary_text = response.content if isinstance(response.content, str) else str(response.content)
        summary_msg = SystemMessage(content=f"[历史摘要] {summary_text}")
        return [summary_msg, *recent]

    except Exception as exc:  # noqa: BLE001
        logger.warning("Context compression LLM call failed (%s); fallback to truncation.", exc)
        return await _fallback_truncate(messages, max_tokens)


async def _fallback_truncate(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Drop earliest messages one by one until total tokens fit within *max_tokens*."""
    result = list(messages)
    while result and _count_tokens_for_messages(result) > max_tokens:
        result = result[1:]
    return result
