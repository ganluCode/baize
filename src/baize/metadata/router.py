"""Metadata API router.

GET /api/v1/metadata            — list available resources
GET /api/v1/metadata/{resource} — field schema for a specific resource
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from baize.core.database import get_db
from baize.core.deps import get_container
from baize.metadata.resolvers import ResolverContext
from baize.metadata.schemas import ResourceMetadata
from baize.user.deps import get_current_user

if TYPE_CHECKING:
    from baize.core.container import Container
    from baize.user.models import UserModel

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.get("")
async def list_resources(
    container: "Container" = Depends(get_container),
) -> dict:
    """Return the list of resources that have metadata definitions."""
    return {"resources": container.metadata_registry.list_resources()}


@router.get("/{resource}", response_model=ResourceMetadata)
async def get_resource_metadata(
    resource: str,
    current_user: "UserModel" = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    container: "Container" = Depends(get_container),
) -> ResourceMetadata:
    """Return field schema and options for the given resource."""
    ctx = ResolverContext(
        user_id=current_user.id,
        db=db,
        provider_factory=container.provider_factory,
        settings=container.config,
    )
    try:
        return await container.metadata_registry.get(resource, ctx)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown resource: {resource}")
