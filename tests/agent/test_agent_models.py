"""Tests for AgentConfig ORM model structure and constraints."""

from sqlalchemy import Boolean, DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from baize.agent.models import AgentConfig


def _col(name: str):
    return AgentConfig.__table__.c[name]


def _index(name: str):
    return next(
        (idx for idx in AgentConfig.__table_args__ if isinstance(idx, Index) and idx.name == name),
        None,
    )


class TestAgentConfigColumns:
    def test_id_is_uuid_primary_key(self):
        col = _col("id")
        assert col.primary_key
        assert isinstance(col.type, UUID)

    def test_user_id_is_uuid_not_null(self):
        col = _col("user_id")
        assert not col.nullable
        assert isinstance(col.type, UUID)

    def test_name_varchar_100_not_null(self):
        col = _col("name")
        assert not col.nullable
        assert isinstance(col.type, String)
        assert col.type.length == 100

    def test_description_varchar_500_nullable(self):
        col = _col("description")
        assert col.nullable
        assert isinstance(col.type, String)
        assert col.type.length == 500

    def test_agent_type_string_not_null(self):
        col = _col("agent_type")
        assert not col.nullable
        assert isinstance(col.type, String)

    def test_is_enabled_boolean_not_null_default_true(self):
        col = _col("is_enabled")
        assert not col.nullable
        assert isinstance(col.type, Boolean)
        assert col.server_default is not None

    def test_prompts_jsonb_nullable(self):
        col = _col("prompts")
        assert col.nullable
        assert isinstance(col.type, JSONB)

    def test_tools_jsonb_nullable(self):
        col = _col("tools")
        assert col.nullable
        assert isinstance(col.type, JSONB)

    def test_model_config_jsonb_nullable(self):
        col = _col("model_config")
        assert col.nullable
        assert isinstance(col.type, JSONB)

    def test_memory_config_jsonb_nullable(self):
        col = _col("memory_config")
        assert col.nullable
        assert isinstance(col.type, JSONB)

    def test_guardrails_jsonb_nullable(self):
        col = _col("guardrails")
        assert col.nullable
        assert isinstance(col.type, JSONB)

    def test_knowledge_config_jsonb_nullable(self):
        col = _col("knowledge_config")
        assert col.nullable
        assert isinstance(col.type, JSONB)

    def test_sub_agents_jsonb_nullable(self):
        col = _col("sub_agents")
        assert col.nullable
        assert isinstance(col.type, JSONB)

    def test_created_at_datetime_not_null(self):
        col = _col("created_at")
        assert not col.nullable
        assert isinstance(col.type, DateTime)

    def test_updated_at_datetime_not_null(self):
        col = _col("updated_at")
        assert not col.nullable
        assert isinstance(col.type, DateTime)


class TestAgentConfigIndexes:
    def test_user_enabled_composite_index_exists(self):
        idx = _index("ix_agent_configs_user_enabled")
        assert idx is not None
        col_names = {col.key for col in idx.columns}
        assert col_names == {"user_id", "is_enabled"}
