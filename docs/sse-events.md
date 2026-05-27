# Baize SSE 事件规范

## 接口

```
POST /api/v1/chat
Authorization: Bearer <jwt 或 api_key>
Content-Type: application/json

{
  "message": "你好",
  "agent_id": "可选，不传用默认智能体",
  "session_id": "可选，不传创建新会话"
}

Response: text/event-stream
```

## 事件类型

### token

LLM 生成的文本片段（逐字输出）。

```
event: token
data: {"content": "你好"}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| content | string | 文本片段，追加到当前回复 |

### thinking

LLM 的思考/推理过程（仅支持推理的模型会产生，如 DeepSeek-R1、o1 等）。

```
event: thinking
data: {"content": "让我分析一下这个问题..."}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| content | string | 思考内容，不计入最终回复 |

前端建议：折叠显示或灰色斜体展示。不关心思考过程可直接忽略此事件。

### tool_call

智能体发起工具调用。

```
event: tool_call
data: {"tool": "save_memory", "args": {"content": "用户喜欢吃火锅"}}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| tool | string | 工具名称 |
| args | object | 调用参数 |

### tool_result

工具调用返回结果。与前一个 `tool_call` 按 `tool` 名称配对。

```
event: tool_result
data: {"tool": "save_memory", "result": "已保存记忆"}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| tool | string | 工具名称（与 tool_call 对应） |
| result | string | 工具返回的文本结果 |

### done

对话完成。包含会话 ID 和消息 ID，前端应保存 `session_id` 用于续聊。

```
event: done
data: {"session_id": "uuid", "message_id": "uuid"}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string (UUID) | 会话 ID，续聊时传回 |
| message_id | string (UUID) | 本轮用户消息的 ID |

### error

执行出错，对话终止。

```
event: error
data: {"message": "对话超时（120秒），请稍后重试。"}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| message | string | 错误描述 |

## 事件顺序

```
典型对话（无工具调用）:
  thinking* → token+ → done

带工具调用:
  thinking* → token* → tool_call → tool_result → token+ → done

多轮工具调用:
  token* → tool_call → tool_result → tool_call → tool_result → token+ → done

出错:
  token* → error
```

- `*` 表示 0 或多次
- `+` 表示 1 或多次
- `thinking` 仅在模型支持推理时出现
- 一次对话只会有一个 `done` 或一个 `error`，二者互斥

## 前端接收示例

```typescript
const response = await fetch('/api/v1/chat', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ message, session_id }),
})

const reader = response.body!.getReader()
const decoder = new TextDecoder()
let buffer = ''

while (true) {
  const { done, value } = await reader.read()
  if (done) break

  buffer += decoder.decode(value, { stream: true })
  const lines = buffer.split('\n')
  buffer = lines.pop() || ''

  let eventType = ''
  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7)
    } else if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6))
      switch (eventType) {
        case 'token':
          // 追加到消息内容
          appendContent(data.content)
          break
        case 'thinking':
          // 显示思考过程（可选）
          appendThinking(data.content)
          break
        case 'tool_call':
          // 显示工具调用状态
          showToolCall(data.tool, data.args)
          break
        case 'tool_result':
          // 更新工具调用结果
          showToolResult(data.tool, data.result)
          break
        case 'done':
          // 保存 session_id 用于续聊
          sessionId = data.session_id
          break
        case 'error':
          showError(data.message)
          break
      }
    }
  }
}
```

## 同步接口

不需要流式的场景用 `/api/v1/chat/sync`，返回完整 JSON：

```
POST /api/v1/chat/sync
→ {"code": 0, "data": {"session_id": "uuid", "message_id": "uuid", "content": "完整回复", "tool_calls": [...]}}
```
