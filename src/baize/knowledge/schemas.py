"""Pydantic request/response schemas for the knowledge module."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseCreateRequest(BaseModel):
    """Request body for creating a new knowledge base."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    settings: dict[str, Any] | None = None


class KnowledgeBaseResponse(BaseModel):
    """Knowledge base representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    description: str | None
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    settings: dict[str, Any] | None
    status: str
    document_count: int
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseListResponse(BaseModel):
    """Paginated list of knowledge bases."""

    items: list[KnowledgeBaseResponse]
    total: int


class KnowledgeDocumentCreateRequest(BaseModel):
    """Request body for creating a document via JSON."""

    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., max_length=2_000_000)
    source_type: Literal["markdown", "raw_text"]
    source_uri: str | None = None
    doc_metadata: dict[str, Any] | None = None


class KnowledgeDocumentResponse(BaseModel):
    """Document representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kb_id: uuid.UUID
    title: str
    source_type: str
    source_uri: str | None
    content_hash: str
    doc_metadata: dict[str, Any] | None
    chunk_count: int
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class KnowledgeDocumentListResponse(BaseModel):
    """Paginated list of knowledge documents."""

    items: list[KnowledgeDocumentResponse]
    total: int


class SearchRequest(BaseModel):
    """Request body for searching a knowledge base."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    include_parents: bool = False


class CitationResponse(BaseModel):
    """A single retrieval result with citation metadata."""

    doc_title: str
    chunk_id: uuid.UUID
    doc_id: uuid.UUID
    section_path: list[str]
    score: float
    sources: list[str]


class SearchResponse(BaseModel):
    """Search results returned by the API."""

    query: str
    results: list[CitationResponse]
    total: int
    latency_ms: int
