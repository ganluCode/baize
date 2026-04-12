"""Metadata registry — central store for resource metadata definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baize.metadata.resolvers import ResolverContext
    from baize.metadata.schemas import ResourceMetadata

ResourceMetadataFactory = Callable[["ResolverContext"], Awaitable["ResourceMetadata"]]


class MetadataRegistry:
    """Stores and retrieves metadata definition factories for each resource."""

    def __init__(self) -> None:
        self._resources: dict[str, ResourceMetadataFactory] = {}

    def register(self, resource_name: str, factory: ResourceMetadataFactory) -> None:
        self._resources[resource_name] = factory

    def list_resources(self) -> list[str]:
        return list(self._resources.keys())

    async def get(self, resource_name: str, context: "ResolverContext") -> "ResourceMetadata":
        factory = self._resources.get(resource_name)
        if factory is None:
            raise KeyError(f"Unknown metadata resource: {resource_name}")
        return await factory(context)
