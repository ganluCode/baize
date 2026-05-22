"""Pydantic configuration models for the Baize memory subsystem."""

from typing import Any

from pydantic import BaseModel


class Mem0LLMConfig(BaseModel):
    """LLM binding for Mem0.

    The ``provider`` field follows the Baize '供应商名/模型ID' convention
    (e.g. ``'doubao/doubao-pro-256k'``).  F-003 will resolve this reference
    against ``LLMProviderManager`` to obtain ``base_url`` and ``api_key``.
    """

    provider: str
    temperature: float | None = None


class Mem0EmbedderConfig(BaseModel):
    """Embedder binding for Mem0.

    The ``provider`` field follows the same '供应商名/模型ID' convention as
    :class:`Mem0LLMConfig`.
    """

    provider: str


class Mem0VectorStoreConfig(BaseModel):
    """Vector-store binding for Mem0."""

    provider: str = "qdrant"
    config: dict[str, Any] = {}


class Mem0Config(BaseModel):
    """Full configuration block passed to the Mem0 adapter (F-003)."""

    llm: Mem0LLMConfig | None = None
    embedder: Mem0EmbedderConfig | None = None
    vector_store: Mem0VectorStoreConfig | None = None


class MemoryConfig(BaseModel):
    """Top-level memory configuration for Baize.

    Contains the provider selection and global behaviour toggles, plus the
    provider-specific sub-config (currently only ``mem0`` is supported).
    """

    provider: str = "mem0"
    auto_recall: bool = True
    shared: bool = True
    mem0: Mem0Config | None = None
