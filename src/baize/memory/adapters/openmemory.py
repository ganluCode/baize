"""OpenMemory REST API adapter for the Baize memory service.

Calls a running OpenMemory server over HTTP. OpenMemory has its own user
system, so Baize maps all requests to a single OpenMemory user (configured
via ``openmemory_user_id``, default ``"default_user"``), and stores the
real Baize user_id in metadata for filtering.

API endpoints (OpenMemory-specific, NOT standard mem0 OSS):
    POST   /api/v1/memories/         — add a memory (body: text, user_id, metadata, app)
    POST   /api/v1/memories/filter   — search/filter memories (body: user_id, search_query)
    GET    /api/v1/memories/         — list memories (?user_id=)
    GET    /api/v1/memories/{id}     — get single memory
    DELETE /api/v1/memories/{id}     — delete a memory
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from baize.memory.interface import MemoryItem, MemoryServiceInterface

logger = logging.getLogger(__name__)


class OpenMemoryAdapter(MemoryServiceInterface):
    """Adapts MemoryServiceInterface to an OpenMemory REST API backend.

    Args:
        base_url: Root URL of the OpenMemory server (e.g. ``http://10.1.0.6:8765``).
        api_key: Optional API key (sent as ``X-API-Key`` header if set).
        openmemory_user_id: The user_id known to OpenMemory (default ``"default_user"``).
        app_name: App name registered in OpenMemory (default ``"baize"``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        openmemory_user_id: str = "default_user",
        app_name: str = "baize",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._openmemory_user_id = openmemory_user_id
        self._app_name = app_name
        self._timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-API-Key"] = api_key
        logger.info("OpenMemoryAdapter initialised: %s (user=%s, app=%s)",
                     self._base_url, openmemory_user_id, app_name)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        return datetime.now(tz=UTC)

    @staticmethod
    def _map_item(raw: dict[str, Any]) -> MemoryItem:
        meta: dict[str, Any] = raw.get("metadata_") or raw.get("metadata") or {}
        return MemoryItem(
            id=str(raw.get("id", "")),
            content=raw.get("content", raw.get("memory", "")),
            user_id=meta.get("baize_user_id", raw.get("user_id", "")),
            agent_id=meta.get("agent_id"),
            session_id=meta.get("session_id"),
            shared=meta.get("shared", True),
            metadata={k: v for k, v in meta.items()
                      if k not in ("baize_user_id", "agent_id", "session_id", "shared")} or None,
            score=raw.get("score"),
            created_at=OpenMemoryAdapter._parse_datetime(raw.get("created_at")),
            updated_at=OpenMemoryAdapter._parse_datetime(raw.get("updated_at", raw.get("created_at"))),
        )

    async def add(
        self,
        content: str,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        shared: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not content:
            raise ValueError("content must not be empty")

        merged_meta: dict[str, Any] = dict(metadata or {})
        merged_meta["baize_user_id"] = user_id
        merged_meta["shared"] = shared
        if agent_id:
            merged_meta["agent_id"] = agent_id
        if session_id:
            merged_meta["session_id"] = session_id

        body: dict[str, Any] = {
            "user_id": self._openmemory_user_id,
            "text": content,
            "metadata": merged_meta,
            "app": self._app_name,
        }

        async with self._client() as client:
            resp = await client.post("/api/v1/memories/", json=body)
            resp.raise_for_status()
            data = resp.json()

        memory_id = str(data.get("id", ""))
        logger.debug("Memory added: id=%s baize_user_id=%s", memory_id, user_id)
        return memory_id

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        body: dict[str, Any] = {
            "user_id": self._openmemory_user_id,
            "search_query": query,
            "size": top_k,
        }

        async with self._client() as client:
            resp = await client.post("/api/v1/memories/filter", json=body)
            resp.raise_for_status()
            data = resp.json()

        raw_items: list[dict[str, Any]] = data.get("items", [])
        items = [self._map_item(r) for r in raw_items]

        # Filter by Baize user_id and visibility
        filtered = []
        for item in items:
            if item.user_id != user_id:
                continue
            if item.shared:
                filtered.append(item)
            elif agent_id and item.agent_id == agent_id:
                filtered.append(item)

        return filtered

    async def get(self, memory_id: str) -> MemoryItem | None:
        async with self._client() as client:
            resp = await client.get(f"/api/v1/memories/{memory_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()

        return self._map_item(data)

    async def delete(self, memory_id: str) -> bool:
        async with self._client() as client:
            resp = await client.delete(f"/api/v1/memories/{memory_id}")
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
        return True

    async def list_all(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MemoryItem]:
        page = (offset // limit) + 1 if limit > 0 else 1

        async with self._client() as client:
            resp = await client.get(
                "/api/v1/memories/",
                params={"user_id": self._openmemory_user_id, "page": page, "size": limit},
            )
            resp.raise_for_status()
            data = resp.json()

        raw_items: list[dict[str, Any]] = data.get("items", [])
        items = [self._map_item(r) for r in raw_items]
        # Filter by Baize user_id
        return [item for item in items if item.user_id == user_id]
