"""Tests for LLM configuration schemas."""

import pytest
from pydantic import ValidationError

from baize.llm.schemas import (
    LLMSettings,
    ModelConfig,
    ProviderConfig,
)


class TestLLMSettingsDuplicateProviderNames:
    def test_duplicate_names_raises_value_error(self) -> None:
        with pytest.raises((ValueError, ValidationError)) as exc_info:
            LLMSettings(
                providers=[
                    ProviderConfig(
                        name="doubao",
                        type="openai",
                        base_url="https://example.com",
                        api_key="key1",
                        models=[ModelConfig(id="model-a", usage=["chat"])],
                    ),
                    ProviderConfig(
                        name="doubao",
                        type="anthropic",
                        base_url="https://example2.com",
                        api_key="key2",
                        models=[ModelConfig(id="model-b", usage=["chat"])],
                    ),
                ]
            )
        assert "doubao" in str(exc_info.value)

    def test_unique_names_succeeds(self) -> None:
        settings = LLMSettings(
            providers=[
                ProviderConfig(
                    name="doubao",
                    type="openai",
                    base_url="https://example.com",
                    api_key="key1",
                    models=[ModelConfig(id="model-a", usage=["chat"])],
                ),
                ProviderConfig(
                    name="claude",
                    type="anthropic",
                    base_url="https://example2.com",
                    api_key="key2",
                    models=[ModelConfig(id="model-b", usage=["reasoning"])],
                ),
            ]
        )
        assert len(settings.providers) == 2

    def test_empty_providers_succeeds(self) -> None:
        settings = LLMSettings(providers=[])
        assert settings.providers == []
