"""LLM management API endpoints for Baize."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from baize.core.deps import get_provider_factory
from baize.llm.provider import ProviderFactory
from baize.user.deps import get_current_user
from baize.user.models import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


class TestRequest(BaseModel):
    """Request body for the LLM connectivity test endpoint."""

    provider: str
    model: str


class TestResponse(BaseModel):
    """Response body for the LLM connectivity test endpoint."""

    success: bool
    latency_ms: int | None = None
    error: str | None = None


@router.get("/providers")
async def list_providers(
    _current_user: UserModel = Depends(get_current_user),
    factory: ProviderFactory = Depends(get_provider_factory),
) -> list[dict[str, Any]]:
    """Return all configured LLM providers with their availability status.

    Sensitive fields (api_key, base_url) are excluded from the response.

    Returns:
        List of provider info dicts with name, type, status, and models.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    return factory.list_providers()


@router.post("/test")
async def test_provider(
    body: TestRequest,
    _current_user: UserModel = Depends(get_current_user),
    factory: ProviderFactory = Depends(get_provider_factory),
) -> TestResponse:
    """Test connectivity to a specific LLM provider/model by sending a ping message.

    Returns a business-level result (HTTP 200 always); errors are encoded in the
    response body rather than as HTTP error status codes.

    Args:
        body: Provider name and model ID to test.

    Returns:
        TestResponse with success=True and latency_ms on success,
        or success=False and error message on failure.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    try:
        model = factory.get_chat_model(body.provider, body.model)
        start = time.monotonic()
        await model.ainvoke([HumanMessage(content="ping")])
        latency_ms = int((time.monotonic() - start) * 1000)
        return TestResponse(success=True, latency_ms=latency_ms)
    except Exception as exc:
        logger.warning(
            "LLM test failed for provider='%s' model='%s': %s",
            body.provider,
            body.model,
            exc,
        )
        return TestResponse(success=False, error=str(exc))
