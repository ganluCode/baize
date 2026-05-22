"""Metadata definition for tasks resource."""

from __future__ import annotations

from baize.metadata.resolvers import ResolverContext
from baize.metadata.schemas import FieldDescriptor, FieldGroup, FieldOption, ResourceMetadata


async def build_tasks_metadata(ctx: ResolverContext) -> ResourceMetadata:
    return ResourceMetadata(
        resource="tasks",
        groups=[
            FieldGroup(
                key="basic",
                label="任务信息",
                sort=0,
                fields=[
                    FieldDescriptor(key="title", label="标题", type="text", required=True, constraints={"min_length": 1, "max_length": 200}),
                    FieldDescriptor(key="content", label="详细描述", type="textarea"),
                    FieldDescriptor(key="priority", label="优先级", type="select", default="medium", options=[
                        FieldOption(value="low", label="低"),
                        FieldOption(value="medium", label="中"),
                        FieldOption(value="high", label="高"),
                    ]),
                    FieldDescriptor(key="status", label="状态", type="select", default="todo", options=[
                        FieldOption(value="todo", label="待办"),
                        FieldOption(value="in_progress", label="进行中"),
                        FieldOption(value="done", label="已完成"),
                    ]),
                    FieldDescriptor(key="due_date", label="截止日期", type="date"),
                    FieldDescriptor(key="source", label="来源", type="select", default="manual", options=[
                        FieldOption(value="manual", label="手动创建"),
                        FieldOption(value="agent", label="智能体创建"),
                    ]),
                ],
            ),
        ],
    )
