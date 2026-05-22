"""Pydantic configuration models for the LLM layer."""

from typing import Literal

from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    """Configuration for a single model within a provider."""

    id: str
    usage: list[str]  # valid values: chat, reasoning, embedding, image


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider (OpenAI-compatible or Anthropic)."""

    name: str
    type: Literal["openai", "anthropic"]
    base_url: str
    api_key: str
    models: list[ModelConfig]


class AgentModelConfig(BaseModel):
    """Per-agent model bindings. Format: '供应商名/模型ID' (e.g. 'doubao/doubao-pro-256k')."""

    chat: str | None = None
    reasoning: str | None = None
    embedding: str | None = None


class AgentConfig(BaseModel):
    """Configuration for a named agent, including its model bindings."""

    name: str
    models: AgentModelConfig


class LLMSettings(BaseModel):
    """Top-level LLM configuration. Provider names must be unique."""

    providers: list[ProviderConfig]

    @model_validator(mode="after")
    def validate_unique_provider_names(self) -> "LLMSettings":
        names = [p.name for p in self.providers]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"Duplicate provider name: '{name}'")
            seen.add(name)
        return self
