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


# ---------------------------------------------------------------------------
# Helpers for F-004 tests
# ---------------------------------------------------------------------------


def _make_adapter() -> Mem0Adapter:
    """Create a Mem0Adapter with a mocked mem0 Memory client."""
    factory = _make_factory()
    config = _make_config(llm_provider="doubao/doubao-pro-256k")
    with patch("baize.memory.adapters.mem0.Memory") as MockMemory:
        MockMemory.from_config.return_value = MagicMock()
        adapter = Mem0Adapter(config=config, llm_provider=factory)
    return adapter


def _make_mem0_item(
    id: str = "mem-001",
    memory: str = "Test content",
    user_id: str = "user-1",
    agent_id: str | None = None,
    metadata: dict | None = None,
    created_at: str = "2026-01-01T00:00:00Z",
    updated_at: str = "2026-01-01T00:00:00Z",
) -> dict:
    """Build a dict resembling a mem0 API result item."""
    item: dict = {
        "id": id,
        "memory": memory,
        "user_id": user_id,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if agent_id is not None:
        item["agent_id"] = agent_id
    if metadata is not None:
        item["metadata"] = metadata
    return item


# ---------------------------------------------------------------------------
# Tests: add()
# ---------------------------------------------------------------------------


class TestMem0AdapterAdd:
    async def test_empty_content_raises_value_error(self) -> None:
        """add() with empty string raises ValueError without calling mem0 API."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        adapter._client = mock_client

        with pytest.raises(ValueError, match="content"):
            await adapter.add(content="", user_id="user-1")
        mock_client.add.assert_not_called()

    async def test_add_calls_client_with_user_id(self) -> None:
        """add() calls mem0_client.add() with user_id and returns memory_id string."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.add.return_value = {
            "results": [{"id": "mem-abc", "memory": "hello", "event": "ADD"}]
        }
        adapter._client = mock_client

        result = await adapter.add(content="hello", user_id="user-1")

        assert result == "mem-abc"
        mock_client.add.assert_called_once()
        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs.get("user_id") == "user-1"

    async def test_add_stores_session_id_and_shared_in_metadata(self) -> None:
        """add() passes session_id and shared flag into metadata."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.add.return_value = {"results": [{"id": "mem-xyz", "memory": "hi"}]}
        adapter._client = mock_client

        await adapter.add(
            content="hi",
            user_id="user-1",
            session_id="sess-99",
            shared=False,
            agent_id="agent-7",
        )

        call_kwargs = mock_client.add.call_args.kwargs
        meta = call_kwargs.get("metadata", {})
        assert meta.get("session_id") == "sess-99"
        assert meta.get("shared") is False

    async def test_add_returns_string_memory_id(self) -> None:
        """Return value of add() is always a str."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.add.return_value = {"results": [{"id": "123", "memory": "x"}]}
        adapter._client = mock_client

        result = await adapter.add(content="x", user_id="u")
        assert isinstance(result, str)
        assert result == "123"


# ---------------------------------------------------------------------------
# Tests: get()
# ---------------------------------------------------------------------------


class TestMem0AdapterGet:
    async def test_get_returns_memory_item_when_found(self) -> None:
        """get() maps mem0 result dict to MemoryItem with correct fields."""
        from baize.memory.interface import MemoryItem

        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.get.return_value = _make_mem0_item(
            id="mem-001",
            memory="Remember me",
            user_id="user-1",
            agent_id="agent-1",
            metadata={"session_id": "sess-1", "shared": False},
        )
        adapter._client = mock_client

        result = await adapter.get("mem-001")

        assert isinstance(result, MemoryItem)
        assert result.id == "mem-001"
        assert result.content == "Remember me"
        assert result.user_id == "user-1"
        assert result.agent_id == "agent-1"
        assert result.session_id == "sess-1"
        assert result.shared is False

    async def test_get_returns_none_when_not_found(self) -> None:
        """get() returns None when mem0 client returns None."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.get.return_value = None
        adapter._client = mock_client

        result = await adapter.get("nonexistent")
        assert result is None

    async def test_get_shared_defaults_to_true_when_not_in_metadata(self) -> None:
        """get() defaults shared=True if not present in metadata."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.get.return_value = _make_mem0_item(metadata=None)
        adapter._client = mock_client

        result = await adapter.get("mem-001")
        assert result is not None
        assert result.shared is True


# ---------------------------------------------------------------------------
# Tests: delete()
# ---------------------------------------------------------------------------


class TestMem0AdapterDelete:
    async def test_delete_returns_true_on_success(self) -> None:
        """delete() returns True when mem0 client deletes successfully."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.delete.return_value = {"message": "Memory deleted successfully!"}
        adapter._client = mock_client

        result = await adapter.delete("mem-001")
        assert result is True

    async def test_delete_returns_false_when_not_found(self) -> None:
        """delete() returns False when mem0 raises ValueError for missing ID."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.delete.side_effect = ValueError("Memory with id mem-999 not found")
        adapter._client = mock_client

        result = await adapter.delete("mem-999")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: list_all()
# ---------------------------------------------------------------------------


class TestMem0AdapterListAll:
    async def test_list_all_returns_list_of_memory_items(self) -> None:
        """list_all() maps mem0 results to list[MemoryItem]."""
        from baize.memory.interface import MemoryItem

        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.get_all.return_value = {
            "results": [
                _make_mem0_item(id="m1", memory="first"),
                _make_mem0_item(id="m2", memory="second"),
            ]
        }
        adapter._client = mock_client

        results = await adapter.list_all(user_id="user-1")

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, MemoryItem) for r in results)
        assert results[0].id == "m1"
        assert results[1].id == "m2"

    async def test_list_all_returns_empty_list_when_no_results(self) -> None:
        """list_all() returns [] when mem0 returns no results."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.get_all.return_value = {"results": []}
        adapter._client = mock_client

        results = await adapter.list_all(user_id="user-1")
        assert results == []

    async def test_list_all_passes_user_id_to_client(self) -> None:
        """list_all() calls get_all with user_id kwarg."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.get_all.return_value = {"results": []}
        adapter._client = mock_client

        await adapter.list_all(user_id="user-42", limit=10, offset=0)

        call_kwargs = mock_client.get_all.call_args.kwargs
        assert call_kwargs.get("user_id") == "user-42"

    async def test_list_all_respects_limit_and_offset(self) -> None:
        """list_all() applies offset slicing and respects limit."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.get_all.return_value = {
            "results": [
                _make_mem0_item(id=f"m{i}", memory=f"item {i}") for i in range(5)
            ]
        }
        adapter._client = mock_client

        results = await adapter.list_all(user_id="user-1", limit=2, offset=2)

        assert len(results) == 2
        assert results[0].id == "m2"
        assert results[1].id == "m3"


# ---------------------------------------------------------------------------
# Helpers for search tests
# ---------------------------------------------------------------------------


def _make_search_item(
    id: str = "mem-001",
    memory: str = "Test content",
    user_id: str = "user-1",
    agent_id: str | None = None,
    shared: bool = True,
    score: float = 0.9,
) -> dict:
    """Build a dict resembling a mem0 search result item."""
    item: dict = {
        "id": id,
        "memory": memory,
        "user_id": user_id,
        "score": score,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "metadata": {"shared": shared},
    }
    if agent_id is not None:
        item["agent_id"] = agent_id
        item["metadata"]["agent_id"] = agent_id
    return item


# ---------------------------------------------------------------------------
# Tests: search()
# ---------------------------------------------------------------------------


class TestMem0AdapterSearch:
    async def test_search_calls_client_with_query_and_user_id(self) -> None:
        """search() calls mem0_client.search() with query and user_id."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        adapter._client = mock_client

        await adapter.search(query="test query", user_id="user-1")

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args
        # first positional arg is query
        assert call_kwargs.args[0] == "test query"
        assert call_kwargs.kwargs.get("user_id") == "user-1"

    async def test_search_returns_empty_list_when_no_results(self) -> None:
        """search() returns [] when mem0 returns no results."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        adapter._client = mock_client

        results = await adapter.search(query="anything", user_id="user-1")
        assert results == []

    async def test_search_without_agent_id_returns_only_shared(self) -> None:
        """Without agent_id, search() only returns items with shared=True."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                _make_search_item(id="m1", shared=True, score=0.9),
                _make_search_item(id="m2", shared=False, score=0.8),
                _make_search_item(id="m3", shared=True, score=0.7),
            ]
        }
        adapter._client = mock_client

        results = await adapter.search(query="test", user_id="user-1")

        ids = [r.id for r in results]
        assert "m1" in ids
        assert "m3" in ids
        assert "m2" not in ids

    async def test_search_with_agent_id_returns_shared_and_agent_private(self) -> None:
        """With agent_id, search() returns shared=True items + private items of that agent."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                _make_search_item(id="shared-1", shared=True, agent_id=None, score=0.9),
                _make_search_item(id="private-agent1", shared=False, agent_id="agent-1", score=0.8),
                _make_search_item(id="private-agent2", shared=False, agent_id="agent-2", score=0.7),
                _make_search_item(id="shared-2", shared=True, agent_id="agent-1", score=0.6),
            ]
        }
        adapter._client = mock_client

        results = await adapter.search(query="test", user_id="user-1", agent_id="agent-1")

        ids = [r.id for r in results]
        assert "shared-1" in ids
        assert "private-agent1" in ids
        assert "shared-2" in ids
        assert "private-agent2" not in ids  # private to agent-2, not agent-1

    async def test_search_results_sorted_by_score_descending(self) -> None:
        """search() returns results sorted by score descending."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                _make_search_item(id="m1", shared=True, score=0.5),
                _make_search_item(id="m2", shared=True, score=0.9),
                _make_search_item(id="m3", shared=True, score=0.7),
            ]
        }
        adapter._client = mock_client

        results = await adapter.search(query="test", user_id="user-1")

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].id == "m2"

    async def test_search_top_k_limits_results(self) -> None:
        """search() passes top_k to mem0_client.search() to control max results."""
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        adapter._client = mock_client

        await adapter.search(query="test", user_id="user-1", top_k=3)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs.get("limit") == 3

    async def test_search_returns_memory_items(self) -> None:
        """search() returns list[MemoryItem] with correct field mapping."""
        from baize.memory.interface import MemoryItem

        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                _make_search_item(id="m1", memory="relevant fact", shared=True, score=0.95),
            ]
        }
        adapter._client = mock_client

        results = await adapter.search(query="fact", user_id="user-1")

        assert len(results) == 1
        assert isinstance(results[0], MemoryItem)
        assert results[0].id == "m1"
        assert results[0].content == "relevant fact"
        assert results[0].score == pytest.approx(0.95)
