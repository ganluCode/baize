"""Tests for ensure_collection using AsyncQdrantClient(":memory:") in-memory mode."""

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance

from baize.knowledge.ingestion.qdrant_client import VectorDimMismatchError, ensure_collection

COLLECTION = "test_collection"


@pytest.fixture
def mem_client():
    return AsyncQdrantClient(":memory:")


async def test_ensure_collection_create(mem_client: AsyncQdrantClient):
    await ensure_collection(COLLECTION, vector_dim=128, client=mem_client)

    assert await mem_client.collection_exists(COLLECTION)
    info = await mem_client.get_collection(COLLECTION)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        dim = next(iter(vectors_config.values())).size
    else:
        dim = vectors_config.size
    assert dim == 128


async def test_ensure_collection_dim_mismatch(mem_client: AsyncQdrantClient):
    await ensure_collection(COLLECTION, vector_dim=128, client=mem_client)

    with pytest.raises(VectorDimMismatchError):
        await ensure_collection(COLLECTION, vector_dim=256, client=mem_client)


async def test_ensure_collection_idempotent(mem_client: AsyncQdrantClient):
    await ensure_collection(COLLECTION, vector_dim=128, distance=Distance.COSINE, client=mem_client)
    await ensure_collection(COLLECTION, vector_dim=128, distance=Distance.COSINE, client=mem_client)

    info = await mem_client.get_collection(COLLECTION)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        dim = next(iter(vectors_config.values())).size
    else:
        dim = vectors_config.size
    assert dim == 128
