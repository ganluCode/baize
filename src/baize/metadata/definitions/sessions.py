"""Metadata definition for sessions resource."""

from __future__ import annotations

from baize.metadata.resolvers import ResolverContext
from baize.metadata.schemas import FieldDescriptor, FieldGroup, FieldOption, ResourceMetadata


async def build_sessions_metadata(ctx: ResolverContext) -> ResourceMetadata:
    return ResourceMetadata(
        resource="sessions",
        groups=[
            FieldGroup(
                key="basic",
                label="会话信息",
                sort=0,
                fields=[
                    FieldDescriptor(key="title", label="会话标题", type="text", constraints={"max_length": 200}),
                    FieldDescriptor(key="status", label="状态", type="select", default="active", options=[
                        FieldOption(value="active", label="活跃"),
                        FieldOption(value="archived", label="已归档"),
                    ]),
                ],
            ),
        ],
    )
