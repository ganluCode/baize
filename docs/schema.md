# Baize 数据库 Schema

> 基于 SQLAlchemy ORM，PostgreSQL 16，最后更新：2026-04-05
> 无物理外键，关联关系由应用层维护

---

## 表关系总览

```
system_users
  │
  ├── agent_configs    (user_id 逻辑关联)
  │       │
  │       └── sessions       (agent_id 逻辑关联)
  │               │
  │               └── chat_messages  (session_id 逻辑关联)
  │
  └── tasks            (user_id 逻辑关联)
```

---

## 1. system_users — 用户表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | UUID | PK | 用户 ID |
| email | VARCHAR(255) | UNIQUE, NOT NULL | 邮箱 |
| name | VARCHAR(100) | UNIQUE, NOT NULL | 用户名 |
| password | TEXT | NOT NULL | 密码哈希 |
| role | VARCHAR(50) | NOT NULL, DEFAULT 'user' | 角色: user / admin |
| avatar | TEXT | | 头像 URL |
| api_key_hash | TEXT | | API Key 哈希值 |
| preferences | JSONB | | 用户偏好设置 |
| default_agent_id | UUID | | 默认智能体 ID（逻辑关联 agent_configs.id） |
| is_active | BOOLEAN | NOT NULL, DEFAULT true | 是否启用 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

**preferences 结构:**
```json
{
  "communication_style": "friendly",
  "language": "zh",
  "focus_areas": ["编程", "写作"],
  "custom_instructions": "..."
}
```

---

## 2. agent_configs — 智能体配置表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | UUID | PK | 智能体 ID |
| user_id | UUID | NOT NULL | 归属用户 |
| name | VARCHAR(100) | NOT NULL | 名称 |
| description | VARCHAR(500) | | 描述 |
| agent_type | VARCHAR(50) | NOT NULL, DEFAULT 'chat' | 类型（预留: chat/rag/workflow） |
| is_enabled | BOOLEAN | NOT NULL, DEFAULT true | 是否启用 |
| prompts | JSONB | | 分层 prompt 配置（soul / behavior 等） |
| model_config | JSONB | | LLM 路由配置 |
| tools | JSONB | | 分层工具配置 |
| sub_agents | JSONB | | 可委派的子智能体 ID 列表 |
| memory_config | JSONB | | 记忆策略 |
| guardrails | JSONB | | 执行护栏 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

**索引:**
- `ix_agent_configs_user_enabled` (user_id, is_enabled)

### JSONB 字段结构

**prompts** — 分层 prompt
```json
{
  "soul": "你是白泽（Baize），{{ user_name }} 的个人 AI 助理...",
  "behavior": "今天是 {{ current_date }}。## 核心能力..."
}
```
- `soul`: 人设/身份层，极少修改；Jinja2 模板（支持 `user_name`、`current_date`）
- `behavior`: 任务/行为指令层，可按需调整
- 两层都可选；未来可扩展 `examples`、`constraints`、`situational` 等层

**model_config** — LLM 路由
```json
{
  "chat": "openai/gpt-4o",
  "reasoning": "anthropic/claude-opus-4-6"
}
```
- 格式: `"provider_name/model_id"`
- 无配置时回退到全局 default agent

**tools** — 分层工具
```json
{
  "builtin": ["save_memory", "search_memory", "create_task", "list_tasks", "complete_task"],
  "mcp_servers": ["web-search", "filesystem"],
  "skills": ["summarize_url", "deep_search"]
}
```
- `builtin`: 内置工具，启动时通过 @register_tool 注册
- `mcp_servers`: MCP 协议外部工具服务器名称
- `skills`: 多步技能包名称

**sub_agents** — 子智能体
```json
["uuid-translator", "uuid-coder"]
```

**memory_config** — 记忆策略
```json
{
  "auto_recall": true,
  "shared": true,
  "top_k": 5
}
```
- `auto_recall`: 对话时是否自动搜索长期记忆注入上下文
- `shared`: 记忆是否跨智能体共享（false = 仅当前智能体可见）
- `top_k`: 召回记忆条数

**guardrails** — 执行护栏
```json
{
  "max_tool_calls": 10,
  "timeout_seconds": 120,
  "max_tokens_per_turn": 4096
}
```

---

## 3. sessions — 会话表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | UUID | PK | 会话 ID |
| user_id | UUID | NOT NULL | 归属用户 |
| agent_id | UUID | NOT NULL | 归属智能体 |
| title | VARCHAR(200) | | 会话标题（首次对话后自动生成） |
| title_gen_attempts | INTEGER | NOT NULL, DEFAULT 0 | 标题生成尝试次数 |
| status | ENUM | NOT NULL, DEFAULT 'active' | active / archived |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

**索引:**
- `ix_sessions_user_agent_updated` (user_id, agent_id, updated_at)

---

## 4. chat_messages — 聊天消息表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | UUID | PK | 消息 ID |
| session_id | UUID | NOT NULL | 归属会话 |
| user_id | UUID | NOT NULL | 发送者 |
| role | ENUM | NOT NULL | user / assistant / system / tool |
| content | TEXT | NOT NULL | 消息内容 |
| tool_calls | JSONB | | 工具调用详情 |
| tool_name | VARCHAR(100) | | 工具名称（role=tool 时） |
| token_usage | JSONB | | Token 用量统计 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |

**索引:**
- `ix_chat_messages_session_created` (session_id, created_at)

**token_usage 结构:**
```json
{
  "prompt_tokens": 1500,
  "completion_tokens": 800,
  "total_tokens": 2300
}
```

---

## 5. tasks — 任务/清单表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | UUID | PK | 任务 ID |
| user_id | UUID | NOT NULL | 归属用户 |
| title | VARCHAR(200) | NOT NULL | 标题 |
| content | TEXT | | 详细描述 |
| priority | ENUM | NOT NULL, DEFAULT 'medium' | low / medium / high |
| status | ENUM | NOT NULL, DEFAULT 'todo' | todo / in_progress / done |
| due_date | DATE | | 截止日期 |
| source | ENUM | NOT NULL, DEFAULT 'manual' | manual / agent |
| tags | JSONB | | 标签列表 |
| related_memory_id | VARCHAR(200) | | 关联记忆 ID |
| completed_at | TIMESTAMPTZ | | 完成时间 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

**索引:**
- `idx_tasks_user_status_created` (user_id, status, created_at)
- `idx_tasks_user_due_date` (user_id, due_date)

---

## ENUM 类型汇总

| ENUM 名 | 值 | 所属表 |
|---------|-----|-------|
| session_status | active, archived | sessions |
| message_role | user, assistant, system, tool | chat_messages |
| task_priority | low, medium, high | tasks |
| task_status | todo, in_progress, done | tasks |
| task_source | manual, agent | tasks |
