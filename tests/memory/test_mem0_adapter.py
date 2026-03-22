"""Tests for Mem0Adapter.__init__ (F-003).

Covers:
- Correct Mem0 Memory client config construction from MemoryConfig + ProviderFactory
- Provider reference parsing ('供应商名/模型ID' format)
- Clear errors when provider is missing or reference format is invalid
"""

from unittest.mock import MagicMock, patch

import pytest

from baize.llm.provider import ProviderFactory, ProviderNotFoundError, ProviderUnavailableError
from baize.llm.schemas import LLMSettings, ModelConfig, ProviderConfig
from baize.memory.config import Mem0Config, Mem0EmbedderConfig, Mem0LLMConfig, MemoryConfig
from baize.memory.adapters.mem0 import Mem0Adapter, MemoryProviderConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factory(
    name: str = "doubao",
    base_url: str = "https://ark.example.com/v3",
    api_key: str = "sk-test-key",
    models: list[ModelConfig] | None = None,
) -> ProviderFactory:
    if models is None:
        models = [
            ModelConfig(id="doubao-pro-256k", usage=["chat"]),
            ModelConfig(id="text-embedding-3-small", usage=["embedding"]),
        ]
    settings = LLMSettings(
        providers=[
            ProviderConfig(
                name=name,
                type="openai",
                base_url=base_url,
                api_key=api_key,
                models=models,
            )
        ]
    )
    return ProviderFactory(settings)


def _make_config(
    llm_provider: str = "doubao/doubao-pro-256k",
    embedder_provider: str | None = None,
) -> MemoryConfig:
    return MemoryConfig(
        mem0=Mem0Config(
            llm=Mem0LLMConfig(provider=llm_provider),
            embedder=Mem0EmbedderConfig(provider=embedder_provider) if embedder_provider else None,
        )
    )


# ---------------------------------------------------------------------------
# Tests: constructor builds correct Mem0 config
# ---------------------------------------------------------------------------


class TestMem0AdapterInit:
    def test_inherits_memory_service_interface(self) -> None:
        """Mem0Adapter must implement MemoryServiceInterface."""
        from baize.memory.interface import MemoryServiceInterface

        assert issubclass(Mem0Adapter, MemoryServiceInterface)

    def test_llm_config_uses_openai_provider(self) -> None:
        """The 'provider' passed to Mem0 for LLM must be 'openai'."""
        factory = _make_factory()
        config = _make_config(llm_provider="doubao/doubao-pro-256k")

        captured: dict = {}

        def fake_from_config(cfg: dict) -> MagicMock:
            captured.update(cfg)
            return MagicMock()

        with patch("baize.memory.adapters.mem0.Memory") as MockMemory:
            MockMemory.from_config.side_effect = fake_from_config
            Mem0Adapter(config=config, llm_provider=factory)

        assert "llm" in captured
        assert captured["llm"]["provider"] == "openai"

    def test_llm_config_openai_base_url_matches_provider(self) -> None:
        """openai_base_url in LLM config must match the provider's base_url."""
        factory = _make_factory(base_url="https://ark.example.com/v3")
        config = _make_config(llm_provider="doubao/doubao-pro-256k")

        captured: dict = {}

        def fake_from_config(cfg: dict) -> MagicMock:
            captured.update(cfg)
            return MagicMock()

        with patch("baize.memory.adapters.mem0.Memory") as MockMemory:
            MockMemory.from_config.side_effect = fake_from_config
            Mem0Adapter(config=config, llm_provider=factory)

        assert captured["llm"]["config"]["openai_base_url"] == "https://ark.example.com/v3"
        assert captured["llm"]["config"]["model"] == "doubao-pro-256k"

    def test_llm_config_api_key_injected(self) -> None:
        """api_key from the provider must appear in the LLM config."""
        factory = _make_factory(api_key="sk-secret-key")
        config = _make_config(llm_provider="doubao/doubao-pro-256k")

        captured: dict = {}

        def fake_from_config(cfg: dict) -> MagicMock:
            captured.update(cfg)
            return MagicMock()

        with patch("baize.memory.adapters.mem0.Memory") as MockMemory:
            MockMemory.from_config.side_effect = fake_from_config
            Mem0Adapter(config=config, llm_provider=factory)

        assert captured["llm"]["config"]["api_key"] == "sk-secret-key"

    def test_embedder_config_uses_openai_provider(self) -> None:
        """The 'provider' passed to Mem0 for the embedder must be 'openai'."""
        factory = _make_factory()
        config = _make_config(
            llm_provider="doubao/doubao-pro-256k",
            embedder_provider="doubao/text-embedding-3-small",
        )

        captured: dict = {}

        def fake_from_config(cfg: dict) -> MagicMock:
            captured.update(cfg)
            return MagicMock()

        with patch("baize.memory.adapters.mem0.Memory") as MockMemory:
            MockMemory.from_config.side_effect = fake_from_config
            Mem0Adapter(config=config, llm_provider=factory)

        assert "embedder" in captured
        assert captured["embedder"]["provider"] == "openai"
        assert captured["embedder"]["config"]["model"] == "text-embedding-3-small"

    def test_temperature_included_when_set(self) -> None:
        """LLM config temperature is passed through when configured."""
        factory = _make_factory()
        config = MemoryConfig(
            mem0=Mem0Config(
                llm=Mem0LLMConfig(provider="doubao/doubao-pro-256k", temperature=0.3),
            )
        )

        captured: dict = {}

        def fake_from_config(cfg: dict) -> MagicMock:
            captured.update(cfg)
            return MagicMock()

        with patch("baize.memory.adapters.mem0.Memory") as MockMemory:
            MockMemory.from_config.side_effect = fake_from_config
            Mem0Adapter(config=config, llm_provider=factory)

        assert captured["llm"]["config"]["temperature"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestMem0AdapterInitErrors:
    def test_invalid_provider_ref_format_raises_clear_error(self) -> None:
        """Provider ref without '/' raises MemoryProviderConfigError, not ValueError/KeyError."""
        factory = _make_factory()
        config = _make_config(llm_provider="doubao-pro-256k")  # missing slash

        with pytest.raises(MemoryProviderConfigError, match="Invalid provider reference"):
            with patch("baize.memory.adapters.mem0.Memory"):
                Mem0Adapter(config=config, llm_provider=factory)

    def test_unknown_provider_raises_clear_error(self) -> None:
        """Referencing a provider not in config raises MemoryProviderConfigError."""
        factory = _make_factory(name="doubao")
        config = _make_config(llm_provider="unknown-provider/some-model")

        with pytest.raises(MemoryProviderConfigError, match="unknown-provider"):
            with patch("baize.memory.adapters.mem0.Memory"):
                Mem0Adapter(config=config, llm_provider=factory)

    def test_unavailable_provider_raises_clear_error(self) -> None:
        """Unavailable provider (missing API key env var) raises MemoryProviderConfigError."""
        import os

        # Use an env-var reference that is not set
        factory = _make_factory(api_key="${BAIZE_TEST_MISSING_API_KEY_XYZ}")
        # Ensure env var is absent
        os.environ.pop("BAIZE_TEST_MISSING_API_KEY_XYZ", None)

        config = _make_config(llm_provider="doubao/doubao-pro-256k")

        with pytest.raises(MemoryProviderConfigError, match="unavailable"):
            with patch("baize.memory.adapters.mem0.Memory"):
                Mem0Adapter(config=config, llm_provider=factory)

    def test_missing_mem0_config_raises_error(self) -> None:
        """MemoryConfig with no mem0 sub-config raises MemoryProviderConfigError."""
        factory = _make_factory()
        config = MemoryConfig()  # mem0=None

        with pytest.raises(MemoryProviderConfigError, match="mem0"):
            with patch("baize.memory.adapters.mem0.Memory"):
                Mem0Adapter(config=config, llm_provider=factory)

    def test_empty_provider_name_in_ref_raises_error(self) -> None:
        """/model-id (empty provider part) raises MemoryProviderConfigError."""
        factory = _make_factory()
        config = _make_config(llm_provider="/doubao-pro-256k")

        with pytest.raises(MemoryProviderConfigError, match="Invalid provider reference"):
            with patch("baize.memory.adapters.mem0.Memory"):
                Mem0Adapter(config=config, llm_provider=factory)
