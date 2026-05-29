"""Knowledge answer node — generates a cited answer using retrieved chunks.

Injects retrieved chunks into the system prompt and instructs the LLM to
answer using ``[^数字]`` citation markers.  Falls back to a notice about
incomplete information when no chunks are available.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, SystemMessage

from baize.agent.graphs.state import AgentState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

_CITATION_INSTRUCTION = (
    "请根据以下检索资料回答用户问题。"
    "在回答中使用 [^数字] 格式标注引用来源，例如 [^1]、[^2]。\n\n"
    "## 检索资料\n\n"
)

_FALLBACK_INSTRUCTION = (
    "未检索到相关资料，请直接回答但说明信息可能不完整。"
)


def _build_system_content(chunks: list[dict]) -> str:
    if not chunks:
        return _FALLBACK_INSTRUCTION

    refs = "\n\n".join(
        f"[^{i + 1}] {chunk['content']}" for i, chunk in enumerate(chunks)
    )
    return _CITATION_INSTRUCTION + refs


def build_knowledge_answer_node(
    *,
    llm: "BaseLanguageModel",
) -> Callable[[AgentState], Awaitable[dict[str, list[BaseMessage]]]]:
    """Build an async LangGraph node that answers using retrieved knowledge chunks.

    Args:
        llm: LangChain chat model used to generate the answer.

    Returns:
        An async ``(state) -> {"messages": [response]}`` node callable.
    """

    async def knowledge_answer_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages: list[BaseMessage] = list(state.get("messages", []))
        chunks: list[dict] = state.get("retrieved_chunks", [])

        system_content = _build_system_content(chunks)
        knowledge_msg = SystemMessage(content=system_content)

        if messages and isinstance(messages[0], SystemMessage):
            combined = messages[0].content + "\n\n" + system_content
            messages[0] = SystemMessage(content=combined)
        else:
            messages.insert(0, knowledge_msg)

        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    return knowledge_answer_node
