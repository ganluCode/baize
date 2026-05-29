"""Unit tests for agents metadata definition (F-006)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from baize.metadata.schemas import FieldDescriptor, FieldGroup, FieldOption


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.get_model_options = MagicMock(return_value=[])
    ctx.get_builtin_tool_options = MagicMock(return_value=[])
    ctx.get_mcp_server_options = MagicMock(return_value=[])
    ctx.get_user_agent_options = AsyncMock(return_value=[])
    return ctx


@pytest.mark.asyncio
async def test_agent_type_has_knowledge_option_enabled():
    """agent_type 选项含 value=knowledge 且未禁用。"""
    from baize.metadata.definitions.agents import build_agents_metadata

    metadata = await build_agents_metadata(_make_ctx())
    basic_group = next(g for g in metadata.groups if g.key == "basic")
    agent_type_field = next(f for f in basic_group.fields if f.key == "agent_type")
    assert agent_type_field.options is not None
    knowledge_option = next((o for o in agent_type_field.options if o.value == "knowledge"), None)
    assert knowledge_option is not None, "agent_type options must contain value='knowledge'"
    assert knowledge_option.label == "知识检索"
    assert knowledge_option.disabled is False


@pytest.mark.asyncio
async def test_agent_type_no_rag_or_disabled_knowledge_option():
    """agent_type 不再含 value=rag 或 disabled 的知识检索占位选项。"""
    from baize.metadata.definitions.agents import build_agents_metadata

    metadata = await build_agents_metadata(_make_ctx())
    basic_group = next(g for g in metadata.groups if g.key == "basic")
    agent_type_field = next(f for f in basic_group.fields if f.key == "agent_type")
    assert agent_type_field.options is not None
    # no rag
    assert all(o.value != "rag" for o in agent_type_field.options)
    # no disabled knowledge-related option
    assert all(not o.disabled for o in agent_type_field.options if "知识" in o.label)


@pytest.mark.asyncio
async def test_knowledge_config_group_exists():
    """knowledge_config FieldGroup 存在。"""
    from baize.metadata.definitions.agents import build_agents_metadata

    metadata = await build_agents_metadata(_make_ctx())
    kb_group = next((g for g in metadata.groups if g.key == "knowledge_config"), None)
    assert kb_group is not None, "knowledge_config FieldGroup must exist"


@pytest.mark.asyncio
async def test_knowledge_config_visible_when():
    """knowledge_config FieldGroup 的 visible_when 条件为 agent_type=knowledge。"""
    from baize.metadata.definitions.agents import build_agents_metadata

    metadata = await build_agents_metadata(_make_ctx())
    kb_group = next(g for g in metadata.groups if g.key == "knowledge_config")
    assert kb_group.visible_when is not None
    assert kb_group.visible_when.get("agent_type") == "knowledge"


@pytest.mark.asyncio
async def test_knowledge_config_default_kb_id_field():
    """knowledge_config 含 default_kb_id（select 类型，options_source 指向 /api/v1/knowledge-bases）。"""
    from baize.metadata.definitions.agents import build_agents_metadata

    metadata = await build_agents_metadata(_make_ctx())
    kb_group = next(g for g in metadata.groups if g.key == "knowledge_config")
    field = next((f for f in kb_group.fields if f.key == "knowledge_config.default_kb_id"), None)
    assert field is not None, "knowledge_config.default_kb_id field must exist"
    assert field.type == "select"
    assert field.options_source == "/api/v1/knowledge-bases"


@pytest.mark.asyncio
async def test_knowledge_config_top_k_field():
    """knowledge_config 含 top_k（int/number 类型，默认 8，min=1，max=50）。"""
    from baize.metadata.definitions.agents import build_agents_metadata

    metadata = await build_agents_metadata(_make_ctx())
    kb_group = next(g for g in metadata.groups if g.key == "knowledge_config")
    field = next((f for f in kb_group.fields if f.key == "knowledge_config.top_k"), None)
    assert field is not None, "knowledge_config.top_k field must exist"
    assert field.type in ("int", "number")
    assert field.default == 8
    assert field.constraints is not None
    assert field.constraints.get("min") == 1
    assert field.constraints.get("max") == 50


@pytest.mark.asyncio
async def test_knowledge_config_include_parents_field():
    """knowledge_config 含 include_parents（bool 类型，默认 True）。"""
    from baize.metadata.definitions.agents import build_agents_metadata

    metadata = await build_agents_metadata(_make_ctx())
    kb_group = next(g for g in metadata.groups if g.key == "knowledge_config")
    field = next((f for f in kb_group.fields if f.key == "knowledge_config.include_parents"), None)
    assert field is not None, "knowledge_config.include_parents field must exist"
    assert field.type in ("bool", "boolean")
    assert field.default is True


def test_field_group_visible_when_schema():
    """FieldGroup 支持 visible_when 字段。"""
    group = FieldGroup(
        key="test",
        label="Test",
        sort=0,
        fields=[],
        visible_when={"agent_type": "knowledge"},
    )
    assert group.visible_when == {"agent_type": "knowledge"}


def test_field_group_visible_when_defaults_none():
    """FieldGroup.visible_when 默认为 None。"""
    group = FieldGroup(key="test", label="Test", sort=0, fields=[])
    assert group.visible_when is None


def test_field_descriptor_options_source_schema():
    """FieldDescriptor 支持 options_source 字段。"""
    field = FieldDescriptor(
        key="kb_id",
        label="知识库",
        type="select",
        options_source="/api/v1/knowledge-bases",
    )
    assert field.options_source == "/api/v1/knowledge-bases"


def test_field_descriptor_options_source_defaults_none():
    """FieldDescriptor.options_source 默认为 None。"""
    field = FieldDescriptor(key="name", label="名称", type="text")
    assert field.options_source is None
