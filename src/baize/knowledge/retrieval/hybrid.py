"""Hybrid retriever combining BM25 and vector search via Reciprocal Rank Fusion."""

import asyncio
import logging
import uuid
from dataclasses import replace

from baize.knowledge.retrieval.bm25 import BM25Retriever
from baize.knowledge.retrieval.exceptions import VectorRetrievalError
from baize.knowledge.retrieval.types import ScoredChunk
from baize.knowledge.retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)

_RRF_K = 60


class HybridRetriever:
    """Fuses BM25 full-text and vector similarity results using Reciprocal Rank Fusion.

    Args:
        bm25_retriever: BM25 full-text retriever instance.
        vector_retriever: Qdrant vector retriever instance.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
    ) -> None:
        self._bm25 = bm25_retriever
        self._vector = vector_retriever

    async def search(
        self,
        *,
        kb_id: uuid.UUID,
        query: str,
        top_k: int,
        candidate_limit: int,
    ) -> list[ScoredChunk]:
        """Run BM25 + Vector concurrently, merge via RRF, return top_k results.

        Args:
            kb_id: Knowledge base to search within.
            query: Natural language search query.
            top_k: Maximum number of fused results to return.
            candidate_limit: How many candidates each retriever should fetch.

        Returns:
            List of ScoredChunk sorted by RRF score descending, rank is 1-based.

        Raises:
            VectorRetrievalError: When Qdrant is unreachable (propagated from vector path).
        """
        bm25_results, vector_results = await self._gather(kb_id=kb_id, query=query, limit=candidate_limit)

        if not bm25_results and not vector_results:
            return []

        merged = self._merge_rrf(bm25_results, vector_results)
        merged.sort(key=lambda c: c.score, reverse=True)

        final: list[ScoredChunk] = []
        for rank, chunk in enumerate(merged[:top_k], start=1):
            final.append(replace(chunk, rank=rank))
        return final

    async def _gather(
        self,
        *,
        kb_id: uuid.UUID,
        query: str,
        limit: int,
    ) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
        """Concurrently call both retrievers; degrade BM25 failures to empty."""
        bm25_task = asyncio.create_task(self._safe_bm25(kb_id=kb_id, query=query, limit=limit))
        vector_task = asyncio.create_task(self._vector.search(kb_id=kb_id, query=query, limit=limit))

        try:
            vector_results = await vector_task
        except VectorRetrievalError:
            bm25_task.cancel()
            raise

        bm25_results = await bm25_task
        return bm25_results, vector_results

    async def _safe_bm25(self, *, kb_id: uuid.UUID, query: str, limit: int) -> list[ScoredChunk]:
        """Call BM25 retriever; catch non-connection errors and degrade to empty list."""
        try:
            return await self._bm25.search(kb_id=kb_id, query=query, limit=limit)
        except VectorRetrievalError:
            raise
        except Exception:
            logger.warning("BM25 retriever failed, degrading to empty results", exc_info=True)
            return []

    @staticmethod
    def _merge_rrf(
        bm25_results: list[ScoredChunk],
        vector_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Merge two result sets using Reciprocal Rank Fusion (k=60)."""
        candidates: dict[uuid.UUID, ScoredChunk] = {}

        for chunk in bm25_results:
            candidates[chunk.chunk_id] = replace(
                chunk,
                score=1.0 / (_RRF_K + chunk.bm25_rank),
                rank=0,
            )

        for chunk in vector_results:
            cid = chunk.chunk_id
            if cid in candidates:
                existing = candidates[cid]
                rrf_score = 1.0 / (_RRF_K + existing.bm25_rank) + 1.0 / (_RRF_K + chunk.vector_rank)
                candidates[cid] = replace(
                    existing,
                    score=rrf_score,
                    vector_rank=chunk.vector_rank,
                    vector_score=chunk.vector_score,
                    sources=["bm25", "vector"],
                )
            else:
                candidates[cid] = replace(
                    chunk,
                    score=1.0 / (_RRF_K + chunk.vector_rank),
                    rank=0,
                )

        return list(candidates.values())
