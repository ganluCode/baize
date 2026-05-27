# Baize（白泽）

> AI 时代的个人操作系统 — 记忆、任务、学习的统一

Baize 是一个以**深度记忆**为核心的个人 AI 助理平台。它不只是一个聊天机器人，而是一个**越用越懂你**的成长伙伴 — 管理你的记忆、任务和学习，通过持续积累形成复利。

## 项目定位

```
OpenClaw / ChatGPT  →  执行者（替我干活）
Baize               →  成长伙伴（陪我变好，越用越懂我）
```

| 核心理念 | 说明 |
|---------|------|
| 深度记忆 | 不是简单的存取，而是理解、整合、反思、遗忘 |
| AI-native 清单 | 为 AI 设计的任务系统，自动关联记忆、主动管理 |
| 终身学习（规划中） | 知识捕获、间隔复习、知识图谱、盲区识别 |
| 开放协议 | 通过 OpenAI 兼容 API 和 MCP 协议对外暴露能力 |

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     客户端层                              │
│                                                          │
│  自有 Web UI    CherryStudio    OpenClaw    OpenCat     │
│  (Next.js)     (OpenAI 兼容)   (飞书/TG)   (iOS)       │
│      │              │              │          │          │
│      │ 原生 API      └──── OpenAI 兼容协议 ────┘          │
│      ▼                    ▼                              │
└──────┼────────────────────┼──────────────────────────────┘
       │                    │
       ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                     API 层                                │
│                                                          │
│  /api/v1/chat              原生 SSE 流式对话             │
│  /api/v1/agents            智能体 CRUD                   │
│  /api/v1/sessions          会话管理                      │
│  /api/v1/tasks             清单系统                      │
│  /api/v1/metadata          前端表单元数据                 │
│                                                          │
│  /v1/chat/completions      OpenAI Chat 兼容（无状态）     │
│  /v1/responses             OpenAI Responses 兼容（有状态）│
│  /v1/models                模型列表（智能体列表）          │
│  /v1/conversations         会话管理                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   智能体引擎层                             │
│                                                          │
│  AgentService.chat() — 节点管线模式                       │
│    ├─ ContextNode    上下文准备（记忆召回 + Prompt 组装）  │
│    ├─ ReactNode      LangGraph ReAct 循环（LLM + 工具）  │
│    └─ PersistNode    消息持久化                           │
│                                                          │
│  ContextService — 上下文工程                              │
│    ├─ Layer 1a: Soul     人设层                           │
│    ├─ Layer 1b: Behavior 行为层                           │
│    ├─ Layer 2: 用户偏好                                   │
│    ├─ Layer 3: 工具描述                                   │
│    └─ Layer 4: 记忆注入                                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    基础设施层                              │
│                                                          │
│  PostgreSQL    Redis    LLM Providers    OpenMemory      │
│  (数据存储)    (缓存)   (OpenAI/Anthropic) (长期记忆)     │
│                                                          │
│  LangFuse                                                │
│  (可观测性)                                               │
└─────────────────────────────────────────────────────────┘
```

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.12 |
| Web 框架 | FastAPI | 0.115+ |
| Agent 框架 | LangChain + LangGraph | v1 |
| 数据库 | PostgreSQL | 16 |
| 缓存 | Redis | 7 |
| 记忆 | OpenMemory (mem0) | HTTP 适配 |
| 可观测 | LangFuse | v2 SDK |
| LLM | Anthropic / OpenAI 兼容 | 多 provider |

## 项目结构

```
src/baize/
├── agent/                # 智能体模块
│   ├── models.py         # AgentConfig ORM
│   ├── schemas.py        # Pydantic 请求/响应
│   ├── service.py        # AgentConfigService + AgentService
│   ├── router.py         # CRUD API
│   ├── chat_router.py    # 原生对话 API
│   ├── graph.py          # LangGraph ReAct 图定义
│   ├── nodes/            # 执行节点（BaseNode 模板方法模式）
│   │   ├── base.py       # 抽象基类（自动计时 + trace）
│   │   ├── context_node.py
│   │   ├── react_node.py
│   │   └── persist_node.py
│   └── tools/            # 工具注册（@register_tool）
│       ├── memory.py     # save_memory / search_memory
│       └── task.py       # create_task / list_tasks / complete_task
├── context/              # 上下文工程模块
│   ├── service.py        # ContextService（记忆召回 + 历史压缩 + Prompt 组装）
│   └── system_prompt.py  # 分层 Prompt 组装器
├── llm/                  # LLM 多 Provider 路由
│   ├── provider.py       # ProviderFactory（OpenAI / Anthropic）
│   ├── model_router.py   # ModelRouter（agent 级模型路由）
│   └── schemas.py        # Provider / Model 配置
├── memory/               # 记忆服务
│   ├── interface.py      # MemoryServiceInterface（抽象）
│   ├── service.py        # 记忆配置解析
│   └── adapters/
│       ├── openmemory.py # OpenMemory HTTP 适配器（推荐）
│       └── mem0.py       # Mem0 SDK 适配器
├── session/              # 会话 + 消息管理
├── task/                 # AI-native 清单系统
├── user/                 # 用户 + 认证（JWT + API Key）
├── auth/                 # 认证中间件
├── metadata/             # 元数据 API（驱动前端表单）
├── openai_compat/        # OpenAI 兼容层
│   ├── router_models.py      # GET /v1/models
│   ├── router_responses.py   # POST /v1/responses（有状态）
│   ├── router_completions.py # POST /v1/chat/completions（无状态）
│   └── helpers.py            # model 名 → agent 路由
├── core/                 # 配置、DI、数据库、可观测性
│   ├── config.py         # Settings（.env + config.yaml）
│   ├── container.py      # 服务容器（手动 DI）
│   ├── observability.py  # TraceCollector（LangFuse v2）
│   └── database.py       # SQLAlchemy async engine
└── main.py               # FastAPI 应用入口
```

## 快速开始

### 环境要求

- Python 3.12
- PostgreSQL 16
- Redis 7
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
git clone https://github.com/yourname/baize.git
cd baize
uv sync
```

### 配置

```bash
# 复制环境变量模板
cp .env .env.local

# 编辑 .env.local，填入你的配置
# - DATABASE_URL, REDIS_URL
# - ADMIN_API_KEY, SECRET_KEY, ADMIN_PASSWORD
# - LLM Provider keys (ANTHROPIC_API_KEY 等)
```

LLM Provider 配置在 `config/config.yaml`：

```yaml
llm:
  providers:
    - name: anthropic
      type: anthropic
      base_url: "https://api.anthropic.com"
      api_key: "${ANTHROPIC_API_KEY}"
      models:
        - id: claude-sonnet-4-6
          usage: [chat, reasoning]

agents:
  - name: default
    models:
      chat: "anthropic/claude-sonnet-4-6"
```

### 初始化数据库

```bash
make migrate
```

### 启动

```bash
# 方式 1：make（推荐）
make dev

# 方式 2：模块启动
uv run python -m baize

# 方式 3：uvicorn
uv run uvicorn baize.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后：
- 健康检查：`http://localhost:8000/health`
- API 文档：`http://localhost:8000/docs`
- 默认管理员：`admin` / `<ADMIN_PASSWORD>`

### 验证

```bash
# 登录
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"login":"admin","password":"你的密码"}'

# 创建默认智能体
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "白泽",
    "prompts": {"soul": "你是白泽，用户的个人 AI 助理。"},
    "set_as_default": true
  }'

# 对话
curl -X POST http://localhost:8000/api/v1/chat/sync \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'
```

## 智能体配置

每个智能体 = 一份配置，通过 API 创建和管理：

```json
{
  "name": "白泽",
  "prompts": {
    "soul": "人设层 — 你是谁",
    "behavior": "行为层 — 你做什么"
  },
  "model_config": {
    "chat": "anthropic/claude-sonnet-4-6"
  },
  "tools": {
    "builtin": ["save_memory", "search_memory", "create_task"],
    "mcp_servers": [],
    "skills": []
  },
  "memory_config": {
    "auto_recall": true,
    "shared": true,
    "top_k": 5
  },
  "guardrails": {
    "max_tool_calls": 10,
    "timeout_seconds": 120
  }
}
```

## API 概览

### 原生 API（自有前端用）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/login` | 登录（JWT） |
| POST | `/api/v1/chat` | SSE 流式对话 |
| POST | `/api/v1/chat/sync` | 同步对话 |
| CRUD | `/api/v1/agents` | 智能体管理 |
| GET | `/api/v1/agents/{id}/sessions` | 会话列表 |
| GET | `/api/v1/sessions/{id}/messages` | 消息历史（游标分页） |
| CRUD | `/api/v1/tasks` | 清单管理 |
| GET | `/api/v1/metadata/{resource}` | 表单元数据 |

### OpenAI 兼容 API（第三方客户端）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/models` | 模型列表（= 智能体列表） |
| POST | `/v1/chat/completions` | Chat 对话（无状态） |
| POST | `/v1/responses` | Responses 对话（有状态） |
| POST | `/v1/conversations` | 创建会话 |

### SSE 事件类型

对话流式返回以下事件（详见 [docs/sse-events.md](docs/sse-events.md)）：

| 事件 | 说明 |
|------|------|
| `token` | LLM 文本输出（逐字） |
| `thinking` | LLM 推理过程（仅推理模型） |
| `tool_call` | 工具调用开始 |
| `tool_result` | 工具调用结果 |
| `done` | 对话完成（含 session_id） |
| `error` | 错误 |

## 认证

支持两种认证方式，通过同一个 `Authorization` header：

```
Authorization: Bearer <jwt>       ← 人类用户登录
Authorization: Bearer <api_key>   ← 客户端/服务（OpenClaw、CherryStudio 等）
```

认证优先级：JWT → X-API-Key header → Bearer token 当 API Key。

## 可观测性

使用 LangFuse v2 追踪每次对话，通过 `TraceCollector` 模式自动上报：

```
Trace: chat
  ├─ Span: retrieval:context  (记忆召回)
  ├─ Generation: llm          (LLM 调用，含 prompt/token/延迟)
  ├─ Span: tool:save_memory   (工具调用)
  └─ Tags: [agent:白泽, tool_use, slow]
```

每个节点（ContextNode / ReactNode / PersistNode）自动上报 span，业务代码不直接接触 LangFuse。

## 第三方客户端对接

### CherryStudio / OpenCat

```
API Base URL: http://your-server:8000/v1
API Key: bz-xxx
Model: baize-白泽
```

### OpenClaw（飞书/Telegram 等）

```yaml
# OpenClaw LLM provider 配置
providers:
  baize:
    baseUrl: http://baize:8000/v1
    apiKey: bz-xxx
    api: openai-completions
    models:
      - id: baize-白泽
```

通过 OpenClaw bindings 把飞书群/私聊映射到不同智能体。

### Open WebUI

```yaml
# docker-compose.yaml
OPENAI_API_BASE_URL: http://baize:8000/v1
OPENAI_API_KEY: bz-xxx
ENABLE_RESPONSES_API_STATEFUL: "true"
```

## Docker 部署

```bash
# 启动 PostgreSQL + Redis + Open WebUI
docker compose up -d

# Baize 后端（开发模式本地跑）
make dev
```

## 开发

```bash
make test           # 跑测试
make lint           # 代码检查
make format         # 代码格式化
make migrate        # 数据库迁移
make export-openapi # 导出 OpenAPI 文档
```

## Roadmap

- [x] 智能体 CRUD + 分层 prompts（soul/behavior）
- [x] LangGraph ReAct 执行循环
- [x] OpenAI 兼容层（/v1/chat/completions + /v1/responses）
- [x] 上下文工程模块（ContextService）
- [x] 节点管线模式（BaseNode + TraceCollector）
- [x] OpenMemory 记忆适配
- [x] LangFuse v2 可观测性
- [x] 前端元数据 API
- [x] Reasoning/thinking 支持
- [ ] MCP Server 接口（对外暴露记忆/清单能力）
- [ ] 记忆系统深化（提取策略、反思、用户画像演化）
- [ ] AI-native 清单（自动关联记忆、主动提醒）
- [ ] 终身学习系统（知识捕获、间隔复习、知识图谱）
- [ ] MCP 工具调用（接入外部 MCP 服务器）

## 相关项目

| 项目 | 定位 | 关系 |
|------|------|------|
| [Portal](https://github.com/yourname/portal) | 统一 Web 后台 | Baize 的管理界面 |
| [Diting（谛听）](https://github.com/yourname/diting) | 代码分析 | 独立工具 |
| [OpenClaw](https://github.com/openclaw/openclaw) | 多平台 AI 助手 | 互补：OpenClaw 管通道，Baize 管记忆 |

## License

MIT
