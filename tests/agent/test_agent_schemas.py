"""Tests for AgentConfig Pydantic schemas validation rules."""

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from baize.agent.schemas import AgentCreate, AgentResponse, AgentUpdate


class TestAgentCreateValidation:
    def test_empty_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentCreate(name="", system_prompt="You are helpful.", tools=[])

    def test_name_too_long_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentCreate(name="x" * 101, system_prompt="You are helpful.", tools=[])

    def test_valid_minimal_create(self):
        schema = AgentCreate(name="My Agent", system_prompt="You are helpful.", tools=[])
        assert schema.name == "My Agent"
        assert schema.system_prompt == "You are helpful."
        assert schema.tools == []

    def test_model_config_alias_accepted_in_json(self):
        """model_config JSON key should map to llm_config attribute."""
        schema = AgentCreate.model_validate(
            {"name": "Agent", "system_prompt": "Hi", "tools": [], "model_config": {"model": "gpt-4"}}
        )
        assert schema.llm_config == {"model": "gpt-4"}

    def test_optional_fields_default_to_none(self):
        schema = AgentCreate(name="Agent", system_prompt="Hi", tools=["save_memory"])
        assert schema.description is None
        assert schema.llm_config is None
        assert schema.auto_memory_recall is None
        assert schema.shared_memory is None


class TestAgentUpdateAllFieldsOptional:
    def test_empty_update_is_valid(self):
        """All fields optional — empty body is valid for PATCH."""
        schema = AgentUpdate()
        assert schema.name is None
        assert schema.system_prompt is None
        assert schema.tools is None

    def test_empty_name_in_update_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentUpdate(name="")

    def test_partial_update_with_name_only(self):
        schema = AgentUpdate(name="New Name")
        assert schema.name == "New Name"
        assert schema.system_prompt is None


class TestAgentResponseFromAttributes:
    def _make_orm_like(self, **overrides):
        """Return a simple object mimicking an AgentConfig ORM instance."""

        class FakeAgentConfig:
            id = uuid.uuid4()
            user_id = uuid.uuid4()
            name = "Default Agent"
            description = None
            system_prompt = "You are Baize."
            tools = ["save_memory"]
            model_config = None  # ORM column named model_config
            auto_memory_recall = True
            shared_memory = False
            is_default = True
            created_at = datetime(2026, 1, 1, tzinfo=UTC)
            updated_at = datetime(2026, 1, 1, tzinfo=UTC)

        obj = FakeAgentConfig()
        for key, val in overrides.items():
            setattr(obj, key, val)
        return obj

    def test_from_orm_object(self):
        orm_obj = self._make_orm_like()
        response = AgentResponse.model_validate(orm_obj)
        assert response.name == "Default Agent"
        assert response.is_default is True
        assert response.tools == ["save_memory"]

    def test_model_config_column_mapped_to_llm_config(self):
        orm_obj = self._make_orm_like(model_config={"temperature": 0.7})
        response = AgentResponse.model_validate(orm_obj)
        assert response.llm_config == {"temperature": 0.7}

    def test_response_contains_all_required_fields(self):
        orm_obj = self._make_orm_like()
        response = AgentResponse.model_validate(orm_obj)
        assert isinstance(response.id, uuid.UUID)
        assert isinstance(response.user_id, uuid.UUID)
        assert isinstance(response.created_at, datetime)
        assert isinstance(response.updated_at, datetime)
        assert response.user_id is not None
