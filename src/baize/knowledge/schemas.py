"""Pydantic request/response schemas for the knowledge module."""

# Document and retrieval schemas (KnowledgeDocumentCreate, KnowledgeDocumentRead,
# KnowledgeChunkRead, SearchRequest, SearchResponse) are implemented in F2/F3.

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class KnowledgeBaseCreate(BaseModel):
    """Request body for creating a new knowledge base."""

    name: str
    description: str | None = None
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    settings: dict[str, Any] | None = None


class KnowledgeBaseRead(KnowledgeBaseCreate):
    """Knowledge base representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    status: str
    created_at: datetime
