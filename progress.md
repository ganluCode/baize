# Progress

## P1F8: 清单模块 (Task Module) — F-001 through F-011

**Status**: ALL PASSED

Implemented the complete task module in a single pass. 481/481 tests pass.

### What was implemented

| Task | Description | Status |
|------|-------------|--------|
| F-001 | Task ORM model (TaskPriority/TaskStatus/TaskSource enums, composite indexes) | ✅ |
| F-002 | Alembic migration 0004_create_tasks | ✅ |
| F-003 | Pydantic schemas (TaskCreate, TaskUpdate, TaskResponse, TaskListResponse, TaskListQuery) | ✅ |
| F-004 | TaskRepository (CRUD, pagination, due_date NULLS LAST sorting, ILIKE title search) | ✅ |
| F-005 | TaskService (HTTP per-request) + TaskEngineService (agent tool singleton) | ✅ |
| F-006 | 5 REST endpoints; registered in main.py | ✅ |
| F-007 | create_task tool — InjectedState user_id, source=agent | ✅ |
| F-008 | list_tasks tool — status='all' disables filter, formatted output | ✅ |
| F-009 | complete_task tool — task_title fuzzy ILIKE match, candidate list on ambiguity | ✅ |
| F-010 | TaskService unit tests (source enforcement, completed_at transitions, 404/403, isolation) | ✅ |
| F-011 | Task API integration tests against real baize_test DB | ✅ |

### Key design decisions

- `TaskService` (HTTP): per-request, constructor takes `TaskRepository`
- `TaskEngineService` (agent tools): singleton in container, creates sessions per-call from engine
- Agent tools use `InjectedState` for user_id (matching memory tools pattern)
- `complete_task` tool accepts `task_title` (not task_id) with fuzzy ILIKE matching
- Updated `tests/agent/tools/test_task_tools.py` to match new tool interface

**Tests**: 481/481 pass
**Regressions**: None

## F-017: API Integration Tests (rework attempt 2)

**Status**: PASSED

Implementation was already complete. All 19 integration tests pass:
- `test_agent_api.py`: Agent CRUD (POST 201, GET list, PUT update, DELETE 204, 404 for other user's agents)
- `test_chat_api.py`: SSE chat endpoint (text/event-stream, token+done events, 401 for invalid API key)
- `test_default_agent.py`: New user gets default agent with is_default=True and non-empty system_prompt

**Tests**: 402/402 tests pass across full suite
**Regression**: No regressions

## F-015: Default Agent Prompt + Auto-Init (rework attempt 3)

**Status**: PASSED

Implementation was already complete and `task_list.json` already had `passes: true` set.
All acceptance criteria are met:

- `config/prompts/default_agent.md` exists with 白泽（Baize）persona, `{{ user_name }}` and `{{ current_date }}` variables
- `AgentConfigService.create_default_agent(user_id)` reads the prompt file, sets tools and `is_default=True`
- `UserService.create_user()` calls `create_default_agent(user_id)` after user creation (with error swallowing)
- Tests cover: default tools used, is_default=True enforced, prompt file read, fallback on missing file, UserService integration

**Tests**: All 359 tests pass including F-015 related tests in `tests/agent/test_agent_service.py` and `tests/user/test_user_service.py`
**Regression**: 359/359 tests passed across full suite

The previous rework block reason was "Agent did not report passes: true" — implementation was already complete.

## F-011: Context Compressor (rework attempt 1)

**Status**: PASSED

The implementation was already complete and `task_list.json` already had `passes: true` set.
All acceptance criteria are met:

- `compress_history(messages, llm, max_tokens=8000, keep_turns=6)` returns compressed list
- Returns original list unchanged when total tokens <= max_tokens (no LLM call)
- Keeps last `keep_turns*2` messages, summarises earlier via LLM
- Summary inserted as `SystemMessage(content='[历史摘要] ...')` at list head
- LLM failure fallback: truncates earliest messages, logs warning
- Token counting uses `tiktoken cl100k_base`
- Located in `src/baize/agent/context.py`

**Tests**: 11/11 passed in `tests/agent/test_context.py`
**Regression**: 352/352 tests passed across full suite

The previous rework block reason was "Agent did not report passes: true" — `task_list.json`
already had `passes: true`. No code changes needed.

## F-009: POST /api/v1/chat streaming SSE endpoint

**Status**: PASSED

Implemented the `/api/v1/chat` streaming SSE endpoint with full pre-validation.

### Changes

- **`src/baize/agent/chat_router.py`**: Implemented `POST /api/v1/chat` with:
  - `ChatStreamRequest` schema (message, optional agent_id, optional session_id)
  - `_get_user_model` dependency wrapping `auth.deps.get_current_user` to load UserModel
  - `_safe_chat_stream` async generator wrapping AgentService errors as SSE error events
  - Pre-validation: empty message → 400, no default agent → 404, session not owned → 403
- **`src/baize/agent/service.py`**: Added `AgentConfigService.get_default(user_id)` method
- **`tests/agent/test_chat_stream.py`**: 12 tests covering all acceptance criteria

### Key design decisions

- Uses `auth.deps.get_current_user` (rejects Service Key with 403)
- Pre-validation errors return HTTP status codes (400/403/404), runtime agent errors emit SSE error events
- `_get_user_model` loads full UserModel from DB using auth user_id string
- Session auto-creation: generates a new UUID when session_id is null, AgentService handles creation

**Tests**: 571/571 pass
**Regressions**: None
