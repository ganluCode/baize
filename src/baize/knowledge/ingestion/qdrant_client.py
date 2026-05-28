"""Async Qdrant client singleton and collection management utilities.

Collection naming convention: baize_kb_{kb_id_hex}
  where kb_id_hex is a UUID formatted without hyphens (lowercase hex).
  Example: for kb_id = 550e8400-e29b-41d4-a716-446655440000
           collection_name = "baize_kb_550e8400e29b41d4a716446655440000"
"""

import asyncio
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from baize.core.config import settings

logger = logging.getLogger(__name__)


class QdrantConnectionError(Exception):
    """Raised when connection to Qdrant server fails or an unexpected error occurs."""


class VectorDimMismatchError(Exception):
    """Raised when an existing collection's vector dimension differs from expected."""


_client: AsyncQdrantClient | None = None
_client_lock = asyncio.Lock()


async def get_qdrant_client() -> AsyncQdrantClient:
    """Return the shared AsyncQdrantClient singleton, initializing on first call.

    Reads settings.qdrant_url and settings.qdrant_api_key.
    The API key is never written to logs.

    Raises:
        QdrantConnectionError: If the Qdrant server is unreachable.
    """
    global _client
    if _client is not None:
        return _client

    async with _client_lock:
        if _client is not None:
            return _client

        url = settings.qdrant_url
        # API key intentionally excluded from log output
        logger.debug("Initializing Qdrant client: url=%s grpc_port=%s", url, settings.qdrant_grpc_port)

        init_kwargs: dict = {"url": url}
        if settings.qdrant_api_key:
            init_kwargs["api_key"] = settings.qdrant_api_key
        if settings.qdrant_grpc_port:
            init_kwargs["grpc_port"] = settings.qdrant_grpc_port
            init_kwargs["prefer_grpc"] = True

        try:
            client = AsyncQdrantClient(**init_kwargs)
            await client.get_collections()  # verify connectivity
            _client = client
            logger.info("Qdrant client ready: url=%s", url)
        except Exception as exc:
            raise QdrantConnectionError(f"Cannot connect to Qdrant at {url}") from exc

    return _client


async def ensure_collection(
    collection_name: str,
    vector_dim: int,
    distance: Distance = Distance.COSINE,
    *,
    client: AsyncQdrantClient | None = None,
) -> None:
    """Ensure a Qdrant collection exists with the specified vector configuration.

    Collection naming convention: baize_kb_{kb_id_hex}
    where kb_id_hex is a UUID formatted without hyphens (lowercase hex).
    Example: baize_kb_550e8400e29b41d4a716446655440000

    Args:
        collection_name: Qdrant collection name.
        vector_dim: Expected vector dimensionality.
        distance: Distance metric to use when creating a new collection.
        client: Optional AsyncQdrantClient; uses the singleton if not provided.

    Raises:
        VectorDimMismatchError: Collection exists with a different vector dimension.
        QdrantConnectionError: Cannot reach Qdrant or an unexpected error occurs.
    """
    if client is None:
        client = await get_qdrant_client()

    try:
        exists = await client.collection_exists(collection_name)
    except Exception as exc:
        raise QdrantConnectionError(
            f"Error checking collection '{collection_name}': {exc}"
        ) from exc

    if exists:
        try:
            info = await client.get_collection(collection_name)
        except Exception as exc:
            raise QdrantConnectionError(
                f"Error retrieving collection info for '{collection_name}': {exc}"
            ) from exc

        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            # Named vectors — inspect the first entry
            existing_dim: int = next(iter(vectors_config.values())).size
        else:
            existing_dim = vectors_config.size

        if existing_dim != vector_dim:
            raise VectorDimMismatchError(
                f"Collection '{collection_name}' has dim={existing_dim}, expected dim={vector_dim}"
            )
        logger.debug("Collection '%s' exists with matching dim=%d", collection_name, vector_dim)
    else:
        logger.info(
            "Creating collection '%s' dim=%d distance=%s",
            collection_name,
            vector_dim,
            distance,
        )
        try:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=distance),
            )
        except Exception as exc:
            raise QdrantConnectionError(
                f"Failed to create collection '{collection_name}': {exc}"
            ) from exc
        logger.info("Collection '%s' created", collection_name)
