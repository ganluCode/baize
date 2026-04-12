"""LLM provider factory for Baize.

Wraps LangChain model constructors behind a unified factory that handles:
- Provider/model lookup from configuration
- Lazy instance creation with caching
- Graceful degradation when API key env vars are missing
"""

import logging
import os
import re

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from baize.llm.schemas import LLMSettings, ProviderConfig

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ProviderNotFoundError(KeyError):
    """Raised when the requested provider name is not in configuration."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        super().__init__(f"LLM provider not found: '{provider_name}'")


class ModelNotFoundError(KeyError):
    """Raised when the requested model ID is not registered for a provider."""

    def __init__(self, provider_name: str, model_id: str) -> None:
        self.provider_name = provider_name
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found in provider '{provider_name}'")


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider is marked unavailable due to missing credentials."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        super().__init__(f"LLM provider '{provider_name}' is unavailable (missing API key)")


# ---------------------------------------------------------------------------
# ProviderFactory
# ---------------------------------------------------------------------------


def _resolve_api_key(raw: str) -> str | None:
    """Resolve a potentially env-var-referenced api_key string.

    Returns the resolved key, or None if the env var is absent.
    Plain strings (not matching ``${...}``) are returned as-is.
    """
    match = _ENV_VAR_PATTERN.match(raw)
    if match:
        var_name = match.group(1)
        return os.environ.get(var_name)
    return raw


class ProviderFactory:
    """Creates and caches LangChain model instances for configured LLM providers.

    Args:
        settings: Parsed LLM configuration.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

        # Map provider name → ProviderConfig
        self._providers: dict[str, ProviderConfig] = {p.name: p for p in settings.providers}

        # Track which providers are unavailable due to missing env var
        self._unavailable: set[str] = set()

        # Resolved api keys per provider (populated at init)
        self._api_keys: dict[str, str] = {}

        # Instance caches: (provider_name, model_id) → model instance
        self._chat_cache: dict[tuple[str, str], object] = {}
        self._embedding_cache: dict[tuple[str, str], object] = {}

        self._init_providers()

    def _init_providers(self) -> None:
        """Resolve API keys for all providers; mark unavailable on missing env var."""
        for name, provider in self._providers.items():
            resolved = _resolve_api_key(provider.api_key)
            if resolved is None:
                # Extract env var name for the log message (safe — no actual key value)
                match = _ENV_VAR_PATTERN.match(provider.api_key)
                var_name = match.group(1) if match else "unknown"
                logger.warning(
                    "Provider '%s' is unavailable: environment variable '%s' is not set.",
                    name,
                    var_name,
                )
                self._unavailable.add(name)
            else:
                self._api_keys[name] = resolved

    def _check_provider(self, provider_name: str) -> ProviderConfig:
        """Look up provider config and enforce availability, or raise."""
        if provider_name not in self._providers:
            raise ProviderNotFoundError(provider_name)
        if provider_name in self._unavailable:
            raise ProviderUnavailableError(provider_name)
        return self._providers[provider_name]

    def _check_model(self, provider: ProviderConfig, model_id: str) -> None:
        """Verify that *model_id* is registered for *provider*."""
        model_ids = {m.id for m in provider.models}
        if model_id not in model_ids:
            raise ModelNotFoundError(provider.name, model_id)

    def get_chat_model(self, provider_name: str, model_id: str) -> ChatOpenAI | ChatAnthropic:
        """Return a cached chat model instance.

        Args:
            provider_name: Name of the provider as defined in configuration.
            model_id: Model identifier registered under that provider.

        Returns:
            A ``ChatOpenAI`` or ``ChatAnthropic`` instance.

        Raises:
            ProviderNotFoundError: Provider name not in config.
            ModelNotFoundError: Model ID not registered for the provider.
            ProviderUnavailableError: Provider's API key env var was missing at startup.
        """
        cache_key = (provider_name, model_id)
        if cache_key in self._chat_cache:
            return self._chat_cache[cache_key]  # type: ignore[return-value]

        provider = self._check_provider(provider_name)
        self._check_model(provider, model_id)

        api_key = self._api_keys[provider_name]

        if provider.type == "openai":
            instance = ChatOpenAI(model=model_id, base_url=provider.base_url, api_key=api_key)
        else:
            instance = ChatAnthropic(model=model_id, base_url=provider.base_url, api_key=api_key)  # type: ignore[call-arg]

        self._chat_cache[cache_key] = instance
        return instance  # type: ignore[return-value]

    def list_providers(self) -> list[dict]:
        """Return all configured providers with availability status (no sensitive fields).

        Returns:
            List of dicts with keys: name, type, status, models.
            Sensitive fields (api_key, base_url) are excluded.
        """
        result = []
        for name, provider in self._providers.items():
            status = "unavailable" if name in self._unavailable else "available"
            result.append(
                {
                    "name": name,
                    "type": provider.type,
                    "status": status,
                    "models": [{"id": m.id, "usage": m.usage} for m in provider.models],
                }
            )
        return result

    def get_provider_info(self, provider_name: str) -> tuple[str, str]:
        """Return (base_url, api_key) for a configured provider.

        Args:
            provider_name: Name of the provider as defined in configuration.

        Returns:
            Tuple of (base_url, resolved_api_key).

        Raises:
            ProviderNotFoundError: Provider name not in config.
            ProviderUnavailableError: Provider's API key env var was missing at startup.
        """
        provider = self._check_provider(provider_name)
        return provider.base_url, self._api_keys[provider_name]

    def get_embedding_model(self, provider_name: str, model_id: str) -> OpenAIEmbeddings:
        """Return a cached embedding model instance.

        Only OpenAI-compatible providers are supported for embeddings.

        Args:
            provider_name: Name of the provider as defined in configuration.
            model_id: Model identifier registered under that provider.

        Returns:
            An ``OpenAIEmbeddings`` instance.

        Raises:
            ProviderNotFoundError: Provider name not in config.
            ModelNotFoundError: Model ID not registered for the provider.
            ProviderUnavailableError: Provider's API key env var was missing at startup.
        """
        cache_key = (provider_name, model_id)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]  # type: ignore[return-value]

        provider = self._check_provider(provider_name)
        self._check_model(provider, model_id)

        api_key = self._api_keys[provider_name]

        instance = OpenAIEmbeddings(model=model_id, base_url=provider.base_url, api_key=api_key)

        self._embedding_cache[cache_key] = instance
        return instance  # type: ignore[return-value]
