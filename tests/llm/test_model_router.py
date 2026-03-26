"""Tests for ModelRouter in baize.llm.model_router."""

from unittest.mock import MagicMock, patch

from baize.llm.model_router import ModelRouter
from baize.llm.provider import ProviderFactory
from baize.llm.schemas import AgentConfig, AgentModelConfig, LLMSettings, ModelConfig, ProviderConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(name: str = "doubao", api_key: str = "sk-test") -> ProviderConfig:
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


def _make_factory(*providers: ProviderConfig) -> ProviderFactory:
    settings = LLMSettings(providers=list(providers))
    return ProviderFactory(settings)


def _make_agent(
    name: str,
    chat: str = "doubao/doubao-pro-256k",
    reasoning: str | None = None,
    embedding: str | None = None,
) -> AgentConfig:
    return AgentConfig(
        name=name,
        models=AgentModelConfig(chat=chat, reasoning=reasoning, embedding=embedding),
    )


# ---------------------------------------------------------------------------
# get_chat_model
# ---------------------------------------------------------------------------


class TestGetChatModel:
    def test_resolves_provider_and_model_correctly(self) -> None:
        factory = _make_factory(_make_provider())
        router = ModelRouter(factory, [_make_agent("default")])

        with patch.object(factory, "get_chat_model", wraps=factory.get_chat_model) as mock:
            mock.return_value = MagicMock()
            router.get_chat_model("default")
            mock.assert_called_once_with("doubao", "doubao-pro-256k")

    def test_unknown_agent_falls_back_to_default(self) -> None:
        factory = _make_factory(_make_provider())
        router = ModelRouter(factory, [_make_agent("default")])

        with patch.object(factory, "get_chat_model", wraps=factory.get_chat_model) as mock:
            mock.return_value = MagicMock()
            router.get_chat_model("unknown-agent")
            mock.assert_called_once_with("doubao", "doubao-pro-256k")

    def test_named_agent_used_when_exists(self) -> None:
        factory = _make_factory(_make_provider())
        custom = _make_agent("custom", chat="doubao/doubao-pro-256k")
        default = _make_agent("default", chat="doubao/doubao-pro-256k")
        router = ModelRouter(factory, [default, custom])

        with patch.object(factory, "get_chat_model") as mock:
            mock.return_value = MagicMock()
            router.get_chat_model("custom")
            mock.assert_called_once_with("doubao", "doubao-pro-256k")


# ---------------------------------------------------------------------------
# get_reasoning_model
# ---------------------------------------------------------------------------


class TestGetReasoningModel:
    def test_uses_reasoning_when_configured(self) -> None:
        factory = _make_factory(_make_provider())
        agent = _make_agent("default", reasoning="doubao/doubao-pro-256k")
        router = ModelRouter(factory, [agent])

        with patch.object(factory, "get_chat_model") as mock:
            mock.return_value = MagicMock()
            router.get_reasoning_model("default")
            mock.assert_called_once_with("doubao", "doubao-pro-256k")

    def test_falls_back_to_chat_when_reasoning_is_none(self) -> None:
        factory = _make_factory(_make_provider())
        agent = _make_agent("default", chat="doubao/doubao-pro-256k", reasoning=None)
        router = ModelRouter(factory, [agent])

        with patch.object(factory, "get_chat_model") as mock:
            mock.return_value = MagicMock()
            router.get_reasoning_model("default")
            mock.assert_called_once_with("doubao", "doubao-pro-256k")

    def test_unknown_agent_falls_back_to_default_reasoning(self) -> None:
        factory = _make_factory(_make_provider())
        default = _make_agent("default", chat="doubao/doubao-pro-256k", reasoning="doubao/doubao-pro-256k")
        router = ModelRouter(factory, [default])

        with patch.object(factory, "get_chat_model") as mock:
            mock.return_value = MagicMock()
            router.get_reasoning_model("unknown-agent")
            mock.assert_called_once_with("doubao", "doubao-pro-256k")


# ---------------------------------------------------------------------------
# get_embedding_model
# ---------------------------------------------------------------------------


class TestGetEmbeddingModel:
    def test_uses_embedding_when_configured(self) -> None:
        factory = _make_factory(_make_provider())
        agent = _make_agent("default", embedding="doubao/text-embedding-3-small")
        router = ModelRouter(factory, [agent])

        with patch.object(factory, "get_embedding_model") as mock:
            mock.return_value = MagicMock()
            router.get_embedding_model("default")
            mock.assert_called_once_with("doubao", "text-embedding-3-small")

    def test_falls_back_to_chat_provider_when_embedding_is_none(self) -> None:
        """When embedding not configured, calls get_embedding_model with the chat ref."""
        factory = _make_factory(_make_provider())
        agent = _make_agent("default", chat="doubao/doubao-pro-256k", embedding=None)
        router = ModelRouter(factory, [agent])

        with patch.object(factory, "get_embedding_model") as mock:
            mock.return_value = MagicMock()
            router.get_embedding_model("default")
            mock.assert_called_once_with("doubao", "doubao-pro-256k")

    def test_unknown_agent_falls_back_to_default(self) -> None:
        factory = _make_factory(_make_provider())
        default = _make_agent("default", embedding="doubao/text-embedding-3-small")
        router = ModelRouter(factory, [default])

        with patch.object(factory, "get_embedding_model") as mock:
            mock.return_value = MagicMock()
            router.get_embedding_model("unknown-agent")
            mock.assert_called_once_with("doubao", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Import / smoke test
# ---------------------------------------------------------------------------


def test_importable() -> None:
    from baize.llm.model_router import ModelRouter as MR  # noqa: F401

    assert MR is not None
