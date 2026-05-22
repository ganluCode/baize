"""Tests for ProviderFactory in baize.llm.provider."""

from unittest.mock import MagicMock, patch

import pytest

from baize.llm.provider import (
    ModelNotFoundError,
    ProviderFactory,
    ProviderNotFoundError,
    ProviderUnavailableError,
)
from baize.llm.schemas import LLMSettings, ModelConfig, ProviderConfig


def _make_settings(*providers: ProviderConfig) -> LLMSettings:
    return LLMSettings(providers=list(providers))


def _openai_provider(name: str = "doubao", api_key: str = "sk-test") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        type="openai",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
        models=[
            ModelConfig(id="doubao-pro-256k", usage=["chat"]),
            ModelConfig(id="text-embedding-3-small", usage=["embedding"]),
        ],
    )


def _anthropic_provider(name: str = "claude", api_key: str = "sk-ant-test") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        type="anthropic",
        base_url="https://api.anthropic.com",
        api_key=api_key,
        models=[ModelConfig(id="claude-sonnet-4-6", usage=["chat", "reasoning"])],
    )


class TestProviderFactoryGetChatModel:
    def test_openai_provider_returns_chat_openai(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = factory.get_chat_model("doubao", "doubao-pro-256k")
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
            assert call_kwargs["api_key"] == "sk-test"
            assert call_kwargs["model"] == "doubao-pro-256k"
            assert result is mock_cls.return_value

    def test_anthropic_provider_returns_chat_anthropic(self) -> None:
        settings = _make_settings(_anthropic_provider())
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = factory.get_chat_model("claude", "claude-sonnet-4-6")
            mock_cls.assert_called_once()
            assert result is mock_cls.return_value

    def test_caching_returns_same_instance(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            result1 = factory.get_chat_model("doubao", "doubao-pro-256k")
            result2 = factory.get_chat_model("doubao", "doubao-pro-256k")
            assert result1 is result2
            mock_cls.assert_called_once()  # only constructed once

    def test_nonexistent_provider_raises_provider_not_found(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with pytest.raises(ProviderNotFoundError):
            factory.get_chat_model("nonexistent", "model")

    def test_nonexistent_model_raises_model_not_found(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with pytest.raises(ModelNotFoundError):
            factory.get_chat_model("doubao", "nonexistent-model")

    def test_unavailable_provider_raises_provider_unavailable(self, monkeypatch) -> None:
        # Remove env var so ${MISSING_KEY} cannot be resolved
        monkeypatch.delenv("MISSING_KEY", raising=False)
        settings = _make_settings(_openai_provider(api_key="${MISSING_KEY}"))
        factory = ProviderFactory(settings)

        with pytest.raises(ProviderUnavailableError):
            factory.get_chat_model("doubao", "doubao-pro-256k")

    def test_different_params_different_instances(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.ChatOpenAI") as mock_cls:
            mock_cls.side_effect = [MagicMock(), MagicMock()]
            result1 = factory.get_chat_model("doubao", "doubao-pro-256k")
            result2 = factory.get_chat_model("doubao", "text-embedding-3-small")
            assert result1 is not result2


class TestProviderFactoryGetEmbeddingModel:
    def test_openai_provider_returns_openai_embeddings(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.OpenAIEmbeddings") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = factory.get_embedding_model("doubao", "text-embedding-3-small")
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
            assert call_kwargs["api_key"] == "sk-test"
            assert result is mock_cls.return_value

    def test_embedding_caching_returns_same_instance(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.OpenAIEmbeddings") as mock_cls:
            mock_cls.return_value = MagicMock()
            r1 = factory.get_embedding_model("doubao", "text-embedding-3-small")
            r2 = factory.get_embedding_model("doubao", "text-embedding-3-small")
            assert r1 is r2
            mock_cls.assert_called_once()

    def test_nonexistent_provider_raises(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with pytest.raises(ProviderNotFoundError):
            factory.get_embedding_model("nonexistent", "model")

    def test_nonexistent_model_raises(self) -> None:
        settings = _make_settings(_openai_provider())
        factory = ProviderFactory(settings)

        with pytest.raises(ModelNotFoundError):
            factory.get_embedding_model("doubao", "nonexistent-model")


class TestProviderFactoryUnavailableLogging:
    def test_warning_logged_for_missing_env_var(self, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.delenv("MISSING_KEY", raising=False)
        settings = _make_settings(_openai_provider(api_key="${MISSING_KEY}"))

        with caplog.at_level(logging.WARNING, logger="baize.llm.provider"):
            ProviderFactory(settings)

        # At least one warning logged
        assert any("doubao" in r.message or "MISSING_KEY" in r.message for r in caplog.records)

    def test_api_key_not_in_warning_log(self, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.setenv("REAL_KEY", "super-secret-key-12345")
        # A plain key provider — we want to ensure keys never appear in logs even for valid providers
        settings = _make_settings(_openai_provider(api_key="${REAL_KEY}"))

        with caplog.at_level(logging.WARNING, logger="baize.llm.provider"):
            ProviderFactory(settings)

        for record in caplog.records:
            assert "super-secret-key-12345" not in record.message


class TestProviderFactoryEnvVarResolution:
    def test_env_var_placeholder_resolved(self, monkeypatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "resolved-api-key")
        settings = _make_settings(_openai_provider(api_key="${MY_API_KEY}"))
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            factory.get_chat_model("doubao", "doubao-pro-256k")
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["api_key"] == "resolved-api-key"

    def test_plain_api_key_used_directly(self) -> None:
        settings = _make_settings(_openai_provider(api_key="plain-key"))
        factory = ProviderFactory(settings)

        with patch("baize.llm.provider.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            factory.get_chat_model("doubao", "doubao-pro-256k")
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["api_key"] == "plain-key"
