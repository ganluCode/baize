"""Metadata definition for system_users resource."""

from __future__ import annotations

from baize.metadata.resolvers import ResolverContext
from baize.metadata.schemas import FieldDescriptor, FieldGroup, FieldOption, ResourceMetadata


async def build_users_metadata(ctx: ResolverContext) -> ResourceMetadata:
    return ResourceMetadata(
        resource="users",
        groups=[
            FieldGroup(
                key="basic",
                label="基本信息",
                sort=0,
                fields=[
                    FieldDescriptor(key="name", label="用户名", type="text", required=True, constraints={"min_length": 1, "max_length": 100}),
                    FieldDescriptor(key="email", label="邮箱", type="text", required=True, constraints={"max_length": 255}),
                    FieldDescriptor(key="role", label="角色", type="select", required=True, default="user", options=[
                        FieldOption(value="user", label="普通用户"),
                        FieldOption(value="admin", label="管理员"),
                        FieldOption(value="service", label="服务账号"),
                    ]),
                    FieldDescriptor(key="is_active", label="启用", type="bool", default=True),
                ],
            ),
            FieldGroup(
                key="preferences",
                label="偏好设置",
                sort=1,
                fields=[
                    FieldDescriptor(key="preferences.communication_style", label="沟通风格", type="select", default="friendly", options=[
                        FieldOption(value="friendly", label="友好"),
                        FieldOption(value="concise", label="简洁"),
                        FieldOption(value="formal", label="正式"),
                        FieldOption(value="casual", label="随意"),
                    ]),
                    FieldDescriptor(key="preferences.language", label="语言", type="select", default="zh", options=[
                        FieldOption(value="zh", label="中文"),
                        FieldOption(value="en", label="English"),
                    ]),
                    FieldDescriptor(key="preferences.focus_areas", label="关注领域", type="multi_select", default=[], options=[
                        FieldOption(value="编程", label="编程"),
                        FieldOption(value="写作", label="写作"),
                        FieldOption(value="数据分析", label="数据分析"),
                        FieldOption(value="学习", label="学习"),
                        FieldOption(value="工作", label="工作"),
                        FieldOption(value="生活", label="生活"),
                    ]),
                    FieldDescriptor(key="preferences.custom_instructions", label="自定义指令", description="自定义 AI 回复偏好", type="textarea"),
                ],
            ),
        ],
    )
