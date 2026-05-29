"""Knowledge retrieval node — fetches relevant chunks before LLM answer.

Reads the last HumanMessage as the search query, calls RetrievalService, and
writes the results to ``state["retrieved_chunks"]``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from baize.agent.graphs.state import AgentState

if TYPE_CHECKING:
    from baize.knowledge.service import RetrievalService

logger = logging.getLogger(__name__)


def _extract_text(content: str | list[Any]) -> str:
    """Extract plain text from a message content value.

    Handles both simple strings and multimodal block lists.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _chunk_to_dict(chunk: Any) -> dict:
    return {
        "chunk_id": str(chunk.chunk_id),
        "doc_id": str(chunk.doc_id),
        "section_path": chunk.section_path,
        "score": chunk.score,
        "content": chunk.content,
    }


def build_knowledge_retrieve_node(
    *,
    retrieval_svc: "RetrievalService",
    kb_id: uuid.UUID,
    top_k: int,
    include_parents: bool,
) -> Callable[[AgentState], Awaitable[dict[str, list[dict]]]]:
    """Build an async LangGraph node that retrieves knowledge chunks.

    Args:
        retrieval_svc: Service used to execute hybrid search.
        kb_id: UUID of the knowledge base to search.
        top_k: Maximum number of chunks to return.
        include_parents: When True, parent chunks are appended to results.

    Returns:
        An async ``(state) -> {"retrieved_chunks": [...]}`` node callable.
    """

    async def knowledge_retrieve_node(state: AgentState) -> dict[str, list[dict]]:
        messages = list(state.get("messages", []))
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        if not human_msgs:
            return {"retrieved_chunks": []}

        query = _extract_text(human_msgs[-1].content)
        if not query.strip():
            return {"retrieved_chunks": []}

        try:
            chunks = await retrieval_svc.search(
                kb_id=kb_id,
                query=query,
                top_k=top_k,
                include_parents=include_parents,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "knowledge_retrieve_node: search failed for kb_id=%s query='%s'",
                kb_id,
                query[:80],
                exc_info=True,
            )
            return {"retrieved_chunks": []}

        return {"retrieved_chunks": [_chunk_to_dict(c) for c in chunks]}

    return knowledge_retrieve_node
