"""Metadata definition for agent_configs resource."""

from __future__ import annotations

from baize.metadata.resolvers import ResolverContext
from baize.metadata.schemas import FieldDescriptor, FieldGroup, FieldOption, ResourceMetadata


async def build_agents_metadata(ctx: ResolverContext) -> ResourceMetadata:
    chat_models = ctx.get_model_options(usage="chat")
    reasoning_models = ctx.get_model_options(usage="reasoning")
    builtin_tools = ctx.get_builtin_tool_options()
    mcp_servers = ctx.get_mcp_server_options()
    user_agents = await ctx.get_user_agent_options()

    return ResourceMetadata(
        resource="agents",
        groups=[
            FieldGroup(
                key="basic",
                label="基本信息",
                sort=0,
                fields=[
                    FieldDescriptor(
                        key="name",
                        label="智能体名称",
                        description="智能体的显示名称",
                        type="text",
                        required=True,
                        constraints={"min_length": 1, "max_length": 100},
                    ),
                    FieldDescriptor(
                        key="description",
                        label="描述",
                        description="智能体的功能说明",
                        type="textarea",
                        constraints={"max_length": 500},
                    ),
                    FieldDescriptor(
                        key="agent_type",
                        label="智能体类型",
                        description="智能体的运行模式",
                        type="select",
                        required=True,
                        default="chat",
                        options=[
                            FieldOption(value="chat", label="对话"),
                            FieldOption(value="knowledge", label="知识检索"),
                            FieldOption(value="workflow", label="工作流（计划中）", disabled=True),
                        ],
                    ),
                    FieldDescriptor(
                        key="is_enabled",
                        label="启用",
                        description="是否激活此智能体",
                        type="bool",
                        default=True,
                    ),
                ],
            ),
            FieldGroup(
                key="prompts",
                label="提示词配置",
                sort=1,
                fields=[
                    FieldDescriptor(
                        key="prompts.soul",
                        label="人设层（Soul）",
                        description="定义智能体的身份和性格，支持 Jinja2 变量：{{ user_name }}、{{ current_date }}",
                        type="textarea",
                    ),
                    FieldDescriptor(
                        key="prompts.behavior",
                        label="行为层（Behavior）",
                        description="定义智能体的任务和行为指令，支持 Jinja2 变量",
                        type="textarea",
                    ),
                ],
            ),
            FieldGroup(
                key="model_config",
                label="模型配置",
                sort=2,
                fields=[
                    FieldDescriptor(
                        key="model_config.chat",
                        label="对话模型",
                        description="主对话使用的 LLM，不配置则使用全局默认",
                        type="select",
                        options=chat_models,
                        options_ref="llm_models_chat",
                    ),
                    FieldDescriptor(
                        key="model_config.reasoning",
                        label="推理模型",
                        description="复杂推理场景使用的 LLM",
                        type="select",
                        options=reasoning_models,
                        options_ref="llm_models_reasoning",
                    ),
                ],
            ),
            FieldGroup(
                key="tools",
                label="工具配置",
                sort=3,
                fields=[
                    FieldDescriptor(
                        key="tools.builtin",
                        label="内置工具",
                        description="选择启用的内置工具",
                        type="multi_select",
                        default=[],
                        options=builtin_tools,
                    ),
                    FieldDescriptor(
                        key="tools.mcp_servers",
                        label="MCP 服务",
                        description="选择启用的 MCP 工具服务",
                        type="multi_select",
                        default=[],
                        options=mcp_servers,
                        options_ref="mcp_servers",
                    ),
                    FieldDescriptor(
                        key="tools.skills",
                        label="技能",
                        description="选择启用的技能模块",
                        type="multi_select",
                        default=[],
                        options=[],
                    ),
                ],
            ),
            FieldGroup(
                key="sub_agents",
                label="子智能体",
                sort=4,
                fields=[
                    FieldDescriptor(
                        key="sub_agents",
                        label="可委派的子智能体",
                        description="选择此智能体可以委派任务的其他智能体",
                        type="multi_select",
                        default=[],
                        options=user_agents,
                        options_ref="user_agents",
                    ),
                ],
            ),
            FieldGroup(
                key="memory_config",
                label="记忆策略",
                sort=5,
                fields=[
                    FieldDescriptor(
                        key="memory_config.auto_recall",
                        label="自动召回",
                        description="对话时是否自动检索相关长期记忆",
                        type="bool",
                        default=True,
                    ),
                    FieldDescriptor(
                        key="memory_config.shared",
                        label="共享记忆",
                        description="记忆是否与其他智能体共享",
                        type="bool",
                        default=True,
                    ),
                    FieldDescriptor(
                        key="memory_config.top_k",
                        label="召回数量",
                        description="自动召回时返回的最大记忆条数",
                        type="int",
                        default=5,
                        constraints={"min": 1, "max": 20},
                    ),
                ],
            ),
            FieldGroup(
                key="knowledge_config",
                label="知识检索配置",
                sort=6,
                visible_when={"agent_type": "knowledge"},
                fields=[
                    FieldDescriptor(
                        key="knowledge_config.default_kb_id",
                        label="默认知识库",
                        description="知识检索时使用的知识库",
                        type="select",
                        options_source="/api/v1/knowledge-bases",
                    ),
                    FieldDescriptor(
                        key="knowledge_config.top_k",
                        label="召回数量",
                        description="单次检索返回的最大 chunk 数",
                        type="int",
                        default=8,
                        constraints={"min": 1, "max": 50},
                    ),
                    FieldDescriptor(
                        key="knowledge_config.include_parents",
                        label="包含父节点",
                        description="是否同时返回 chunk 的父级段落以提供更多上下文",
                        type="bool",
                        default=True,
                    ),
                ],
            ),
            FieldGroup(
                key="guardrails",
                label="执行护栏",
                sort=7,
                fields=[
                    FieldDescriptor(
                        key="guardrails.max_tool_calls",
                        label="最大工具调用次数",
                        description="单轮对话允许的最大工具调用次数",
                        type="int",
                        default=10,
                        constraints={"min": 1, "max": 50},
                    ),
                    FieldDescriptor(
                        key="guardrails.timeout_seconds",
                        label="超时时间（秒）",
                        description="单轮对话的最大执行时间",
                        type="int",
                        default=120,
                        constraints={"min": 10, "max": 600},
                    ),
                    FieldDescriptor(
                        key="guardrails.max_tokens_per_turn",
                        label="每轮最大 Token 数",
                        description="每轮对话的输出 token 上限，留空不限制",
                        type="int",
                        default=None,
                        constraints={"min": 100, "max": 32000},
                    ),
                ],
            ),
        ],
    )
