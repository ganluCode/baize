"""LangFuse observability integration for Baize.

Provides a factory function that creates a LangFuse callback handler for
injecting into LangGraph/LangChain invocations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_langfuse_handler(settings=None):  # type: ignore[no-untyped-def]
    """Create a LangFuse :class:`CallbackHandler` for tracing LangGraph runs.

    When :attr:`~baize.core.config.Settings.langfuse_public_key` is not
    configured (``None``), the function returns ``None`` without raising any
    exception, allowing the application to run without LangFuse tracing.

    When configured, the Langfuse global client is initialized with the
    ``public_key``, ``secret_key``, and ``host`` from *settings* and a fresh
    :class:`~langfuse.langchain.CallbackHandler` instance is returned.  Each
    call produces a **new** instance to avoid mixing traces across requests.

    The LangFuse HTTP upload is asynchronous; network errors during tracing
    will not propagate to the caller.

    Args:
        settings: A :class:`~baize.core.config.Settings` instance.  When
            ``None``, :func:`~baize.core.config.get_settings` is called to
            retrieve the singleton.

    Returns:
        A :class:`langfuse.langchain.CallbackHandler` instance when LangFuse
        is configured, or ``None`` otherwise.
    """
    if settings is None:
        from baize.core.config import get_settings

        settings = get_settings()

    if not settings.langfuse_public_key:
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Initialize the global Langfuse client with project credentials.
        # In langfuse v4 this configures the OTEL tracer provider globally.
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

        return CallbackHandler()
    except Exception:
        logger.warning("Failed to create LangFuse handler; tracing disabled.", exc_info=True)
        return None


LangfuseCallbackHandler = Optional[object]
"""Type alias for the optional callback handler returned by :func:`create_langfuse_handler`."""
