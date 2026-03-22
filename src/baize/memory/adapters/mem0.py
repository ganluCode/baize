"""Mem0 adapter for the Baize memory service.

Wraps the ``mem0`` library's ``Memory`` class behind
:class:`~baize.memory.interface.MemoryServiceInterface`, translating between
Baize's provider-reference format (``'供应商名/模型ID'``) and the configuration
dict expected by ``mem0.Memory.from_config()``.
"""

from __future__ import annotations

import logging
from typing import Any

from mem0 import Memory

from baize.llm.provider import ProviderFactory, ProviderNotFoundError, ProviderUnavailableError
from baize.memory.config import MemoryConfig
from baize.memory.interface import MemoryItem, MemoryServiceInterface

logger = logging.getLogger(__name__)


class MemoryProviderConfigError(ValueError):
    """Raised when the memory adapter cannot resolve a required LLM/embedder provider."""


class Mem0Adapter(MemoryServiceInterface):
    """Adapts :class:`~baize.memory.interface.MemoryServiceInterface` to the Mem0 backend.

    Args:
        config: Top-level memory configuration (must include a ``mem0`` sub-config).
        llm_provider: Initialised :class:`~baize.llm.provider.ProviderFactory` used
            to resolve provider references to ``base_url`` / ``api_key`` pairs.

    Raises:
        MemoryProviderConfigError: If the ``mem0`` sub-config is absent, a provider
            reference is malformed, or the referenced provider is not available.
    """

    def __init__(self, config: MemoryConfig, llm_provider: ProviderFactory) -> None:
        if config.mem0 is None:
            raise MemoryProviderConfigError(
                "Mem0Adapter requires a 'mem0' sub-config in MemoryConfig, but none was provided."
            )

        mem0_cfg: dict[str, Any] = {}

        if config.mem0.llm:
            provider_name, model_id = self._parse_provider_ref(config.mem0.llm.provider)
            base_url, api_key = self._resolve_credentials(llm_provider, provider_name, "llm")
            llm_config: dict[str, Any] = {
                "model": model_id,
                "api_key": api_key,
                "openai_base_url": base_url,
            }
            if config.mem0.llm.temperature is not None:
                llm_config["temperature"] = config.mem0.llm.temperature
            mem0_cfg["llm"] = {"provider": "openai", "config": llm_config}

        if config.mem0.embedder:
            provider_name, model_id = self._parse_provider_ref(config.mem0.embedder.provider)
            base_url, api_key = self._resolve_credentials(llm_provider, provider_name, "embedder")
            mem0_cfg["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": model_id,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                },
            }

        if config.mem0.vector_store:
            mem0_cfg["vector_store"] = {
                "provider": config.mem0.vector_store.provider,
                "config": config.mem0.vector_store.config,
            }

        self._client: Memory = Memory.from_config(mem0_cfg)
        logger.debug("Mem0Adapter initialised with config keys: %s", list(mem0_cfg.keys()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_provider_ref(ref: str) -> tuple[str, str]:
        """Parse a ``'供应商名/模型ID'`` string into ``(provider_name, model_id)``.

        Args:
            ref: Provider reference string, e.g. ``'doubao/doubao-pro-256k'``.

        Returns:
            Tuple of ``(provider_name, model_id)``.

        Raises:
            MemoryProviderConfigError: If the string does not contain exactly one
                ``'/'``, or either part is empty.
        """
        parts = ref.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise MemoryProviderConfigError(
                f"Invalid provider reference '{ref}': expected '供应商名/模型ID' format "
                "(e.g. 'doubao/doubao-pro-256k')."
            )
        return parts[0], parts[1]

    @staticmethod
    def _resolve_credentials(
        llm_provider: ProviderFactory,
        provider_name: str,
        role: str,
    ) -> tuple[str, str]:
        """Look up ``base_url`` and ``api_key`` for *provider_name*.

        Args:
            llm_provider: Factory to query.
            provider_name: Name as defined in the LLM configuration.
            role: Human-readable role label (``'llm'`` or ``'embedder'``) used in
                error messages.

        Returns:
            Tuple of ``(base_url, api_key)``.

        Raises:
            MemoryProviderConfigError: With a descriptive message when the provider
                is not found or is unavailable.
        """
        try:
            return llm_provider.get_provider_info(provider_name)
        except ProviderNotFoundError:
            raise MemoryProviderConfigError(
                f"Memory {role} config references provider '{provider_name}', but that provider "
                "is not defined in the LLM configuration. Check your settings."
            ) from None
        except ProviderUnavailableError:
            raise MemoryProviderConfigError(
                f"Memory {role} config references provider '{provider_name}', but that provider "
                "is unavailable (API key environment variable is not set)."
            ) from None

    # ------------------------------------------------------------------
    # MemoryServiceInterface — stubs (implemented in F-004 / F-005)
    # ------------------------------------------------------------------

    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        shared: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError("Implemented in F-004")

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        raise NotImplementedError("Implemented in F-005")

    async def get(self, memory_id: str) -> MemoryItem | None:
        raise NotImplementedError("Implemented in F-004")

    async def delete(self, memory_id: str) -> bool:
        raise NotImplementedError("Implemented in F-004")

    async def list_all(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MemoryItem]:
        raise NotImplementedError("Implemented in F-004")
