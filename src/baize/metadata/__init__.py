"""Metadata module — provides resource field schema API for frontend form rendering."""

from baize.metadata.registry import MetadataRegistry


def create_metadata_registry() -> MetadataRegistry:
    """Build and populate the metadata registry with all resource definitions."""
    from baize.metadata.definitions.agents import build_agents_metadata
    from baize.metadata.definitions.sessions import build_sessions_metadata
    from baize.metadata.definitions.tasks import build_tasks_metadata
    from baize.metadata.definitions.users import build_users_metadata

    registry = MetadataRegistry()
    registry.register("agents", build_agents_metadata)
    registry.register("users", build_users_metadata)
    registry.register("tasks", build_tasks_metadata)
    registry.register("sessions", build_sessions_metadata)
    return registry
