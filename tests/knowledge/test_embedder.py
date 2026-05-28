"""Tests for KnowledgeEmbedder."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
from baize.knowledge.ingestion.qdrant_client import VectorDimMismatchError


@pytest.fixture
def fake_model_and_factory():
    """Return (mock_factory, mock_embedding_model) pair."""
    model = MagicMock()
    factory = MagicMock()
    factory.get_embedding_model.return_value = model
    return factory, model


async def test_embed_query_returns_correct_dim(fake_model_and_factory):
    factory, model = fake_model_and_factory
    model.aembed_query = AsyncMock(return_value=[0.1] * 2048)

    embedder = KnowledgeEmbedder(
        provider_name="test_provider",
        model_id="test_model",
        expected_dim=2048,
        llm_provider_manager=factory,
    )
    result = await embedder.embed_query("hello world")

    assert len(result) == 2048
    factory.get_embedding_model.assert_called_once_with("test_provider", "test_model")


async def test_embed_query_dim_mismatch(fake_model_and_factory):
    factory, model = fake_model_and_factory
    model.aembed_query = AsyncMock(return_value=[0.1] * 512)

    embedder = KnowledgeEmbedder(
        provider_name="test_provider",
        model_id="test_model",
        expected_dim=2048,
        llm_provider_manager=factory,
    )
    with pytest.raises(VectorDimMismatchError):
        await embedder.embed_query("hello world")


async def test_embed_query_dim_check_only_on_first_call(fake_model_and_factory):
    factory, model = fake_model_and_factory
    model.aembed_query = AsyncMock(return_value=[0.1] * 2048)

    embedder = KnowledgeEmbedder(
        provider_name="test_provider",
        model_id="test_model",
        expected_dim=2048,
        llm_provider_manager=factory,
    )
    await embedder.embed_query("first call")
    # Second call with same model (dim already validated) must not re-raise
    result = await embedder.embed_query("second call")
    assert len(result) == 2048


async def test_embed_documents_batch(fake_model_and_factory):
    factory, model = fake_model_and_factory
    model.aembed_documents = AsyncMock(return_value=[[0.1] * 2048] * 3)

    embedder = KnowledgeEmbedder(
        provider_name="test_provider",
        model_id="test_model",
        expected_dim=2048,
        llm_provider_manager=factory,
    )
    results = await embedder.embed_documents(["text1", "text2", "text3"])

    assert len(results) == 3
    assert all(len(v) == 2048 for v in results)
    model.aembed_documents.assert_called_once_with(["text1", "text2", "text3"])


async def test_embed_documents_debug_log(fake_model_and_factory, caplog):
    factory, model = fake_model_and_factory
    model.aembed_documents = AsyncMock(return_value=[[0.1] * 2048] * 2)

    embedder = KnowledgeEmbedder(
        provider_name="test_provider",
        model_id="test_model",
        expected_dim=2048,
        llm_provider_manager=factory,
    )
    with caplog.at_level(logging.DEBUG, logger="baize.knowledge.ingestion.embedder"):
        await embedder.embed_documents(["a", "b"])

    log_messages = " ".join(r.message for r in caplog.records)
    assert "count=2" in log_messages
    assert "latency_ms" in log_messages
    # vector content must NOT be logged
    assert "0.1" not in log_messages


async def test_embed_query_debug_log(fake_model_and_factory, caplog):
    factory, model = fake_model_and_factory
    model.aembed_query = AsyncMock(return_value=[0.1] * 2048)

    embedder = KnowledgeEmbedder(
        provider_name="test_provider",
        model_id="test_model",
        expected_dim=2048,
        llm_provider_manager=factory,
    )
    with caplog.at_level(logging.DEBUG, logger="baize.knowledge.ingestion.embedder"):
        await embedder.embed_query("test")

    log_messages = " ".join(r.message for r in caplog.records)
    assert "latency_ms" in log_messages
    assert "0.1" not in log_messages
