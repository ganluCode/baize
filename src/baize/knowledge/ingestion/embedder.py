"""KnowledgeEmbedder: wraps ProviderFactory for batch and single-query embedding."""

import logging
import time

from baize.knowledge.ingestion.qdrant_client import VectorDimMismatchError
from baize.llm.provider import ProviderFactory

logger = logging.getLogger(__name__)


class KnowledgeEmbedder:
    """Wraps a ProviderFactory embedding model for knowledge ingestion.

    Args:
        provider_name: Provider name as configured in LLM settings.
        model_id: Model identifier under the provider.
        expected_dim: Expected vector dimensionality; validated on first embed_query call.
        llm_provider_manager: ProviderFactory instance used to retrieve the embedding
            model.  Inject a mock here for testing.
    """

    def __init__(
        self,
        provider_name: str,
        model_id: str,
        expected_dim: int,
        *,
        llm_provider_manager: ProviderFactory,
    ) -> None:
        self._provider_name = provider_name
        self._model_id = model_id
        self._expected_dim = expected_dim
        self._llm_provider_manager = llm_provider_manager
        self._dim_validated = False

    def _get_model(self):
        """Return the embedding model from the provider factory."""
        return self._llm_provider_manager.get_embedding_model(self._provider_name, self._model_id)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        model = self._get_model()
        start = time.monotonic()
        vectors = await model.aembed_documents(texts)
        latency_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "embed_documents provider=%s model=%s count=%d latency_ms=%.1f",
            self._provider_name,
            self._model_id,
            len(vectors),
            latency_ms,
        )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        On the first call, validates that the returned vector length matches
        ``expected_dim``.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector for the query.

        Raises:
            VectorDimMismatchError: If the returned vector dimension does not match
                ``expected_dim`` (checked on first call only).
        """
        model = self._get_model()
        start = time.monotonic()
        vector = await model.aembed_query(text)
        latency_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "embed_query provider=%s model=%s latency_ms=%.1f",
            self._provider_name,
            self._model_id,
            latency_ms,
        )

        if not self._dim_validated:
            actual_dim = len(vector)
            if actual_dim != self._expected_dim:
                raise VectorDimMismatchError(
                    f"Embedding model returned dim={actual_dim}, "
                    f"expected dim={self._expected_dim} "
                    f"(provider={self._provider_name}, model={self._model_id})"
                )
            self._dim_validated = True

        return vector
