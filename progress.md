# Progress

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
