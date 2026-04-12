# Baize 项目开发规范

## 可观测性 — TraceCollector 规范

项目使用 LangFuse v2 做可观测性追踪，通过 `TraceCollector` 模式手动上报 trace。
**禁止**使用 langfuse 的 langchain CallbackHandler（v2 SDK 与 langchain v1 不兼容）。

### 使用方式

```python
from baize.core.observability import TraceCollector

# 1. 在流程开始时创建 collector（立即创建 LangFuse trace）
collector = TraceCollector(
    user_id=str(user_id),
    session_id=str(session_id),
    agent_id=str(agent_id),
)
collector.set_input(message)

# 2. 每完成一个节点，立即上报 span（实时写入 LangFuse，不丢数据）
collector.add_llm_span(model=..., input=..., output=..., prompt_tokens=..., completion_tokens=..., latency_ms=...)
collector.add_tool_span(name=..., args=..., result=..., latency_ms=...)
collector.add_retrieval_span(source="memory", query=..., results_count=..., latency_ms=...)
collector.add_sub_agent_span(agent_id=..., agent_name=..., input=..., output=..., latency_ms=...)

# 3. 流程结束时 finalize（更新根 trace 的 output）
collector.finalize(output=full_response)

# 4. 出错时也 finalize（记录错误状态）
collector.finalize(output=partial_response, error="超时")
```

### 规则

1. **业务代码不直接 import langfuse** — 只通过 `TraceCollector` 交互
2. **每个 chat turn 一个 collector** — 在 `AgentService.chat()` 开头创建
3. **span 实时上报** — `add_xxx_span()` 调用时立即写入 LangFuse，进程崩溃也不丢已上报的 span
4. **所有退出路径都要 finalize** — 正常结束、超时、异常，都调 `collector.finalize()`
5. **新增节点类型时**：在 `observability.py` 加对应的 `XxxSpan` dataclass 和 `add_xxx_span` 方法
6. **LangFuse 不可用时** — 所有方法静默跳过，不影响业务流程

### Span 类型

| Span | 方法 | 用途 |
|------|------|------|
| `LLMSpan` | `add_llm_span()` | LLM 调用（模型、token、延迟） |
| `ToolSpan` | `add_tool_span()` | 工具调用（名称、参数、结果） |
| `RetrievalSpan` | `add_retrieval_span()` | 检索操作（记忆召回、RAG、搜索） |
| `SubAgentSpan` | `add_sub_agent_span()` | 子智能体委派 |

## 节点模式 — BaseNode 规范

所有 agent 管线步骤继承 `BaseNode`（模板方法模式），自动完成计时和 trace 上报。

### 当前节点

| 节点 | 文件 | 职责 |
| ---- | ---- | ---- |
| `ContextNode` | `agent/nodes/context_node.py` | 记忆召回 + Prompt 组装 + 历史压缩 |
| `ReactNode` | `agent/nodes/react_node.py` | LangGraph ReAct 循环（流式，报告 LLM/Tool span） |
| `PersistNode` | `agent/nodes/persist_node.py` | 消息持久化 |

### 新增节点规则

1. 继承 `BaseNode`，实现 `_execute(ctx)` 和 `_report(ctx, result, latency_ms)`
2. 业务逻辑只写在 `_execute()` 里，**不碰 tracing**
3. `_report()` 里调 `ctx.collector.add_xxx_span()` 上报
4. 节点间通过 `ctx.results` 字典传递数据
5. 流式节点（如 ReactNode）用 `stream(ctx)` 代替 `run(ctx)`

### 执行管线

```
AgentService.chat()
  │
  ├─ ContextNode.run(ctx)        # 上下文准备（auto-traced）
  ├─ ReactNode.stream(ctx)       # ReAct 循环（LLM/Tool span 实时上报）
  ├─ PersistNode.run(ctx)        # 消息存储（auto-traced）
  └─ collector.finalize()        # 更新根 trace
```

## 技术栈版本约束

| 组件 | 版本 | 原因 |
|------|------|------|
| Python | 3.12.x | pyproject.toml 锁定 |
| langfuse SDK | v2.x（<3.0） | LangFuse server 为 v2.95，SDK v3+ 需要 server v3.125+ |
| langfuse server | v2.95 | 升级需要额外中间件，暂不升级 |
| langchain | v1.x | 当前主线版本 |

## 数据库

- PostgreSQL 16，**无物理外键**，关联关系由应用层维护
- 迁移工具：Alembic（异步模式）
- ORM：SQLAlchemy async

## 记忆服务

- 通过 `OpenMemoryAdapter` HTTP 调用 OpenMemory 服务
- OpenMemory 使用单用户模式（`default_user`），Baize user_id 存在 metadata 里
- 配置项：`.env` 的 `OPENMEMORY_BASE_URL`
