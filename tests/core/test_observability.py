"""Unit tests for baize.core.observability.create_langfuse_handler()."""

from unittest.mock import patch

from baize.core.observability import create_langfuse_handler


class _Settings:
    """Minimal settings stub."""

    def __init__(self, public_key=None, secret_key=None, host="http://langfuse:3000"):
        self.langfuse_public_key = public_key
        self.langfuse_secret_key = secret_key
        self.langfuse_host = host


class TestCreateLangfuseHandlerDisabled:
    """When LangFuse is not configured, the factory must return None safely."""

    def test_returns_none_when_public_key_is_none(self):
        result = create_langfuse_handler(_Settings(public_key=None))
        assert result is None

    def test_returns_none_when_public_key_is_empty_string(self):
        result = create_langfuse_handler(_Settings(public_key=""))
        assert result is None

    def test_no_exception_raised_when_unconfigured(self):
        # Must not raise regardless of environment state
        create_langfuse_handler(_Settings(public_key=None))


class TestCreateLangfuseHandlerEnabled:
    """When LangFuse is configured, create_langfuse_handler returns None (compatibility stub)."""

    def test_returns_none_regardless_of_credentials(self):
        """create_langfuse_handler is a compatibility stub — always returns None."""
        settings = _Settings(public_key="pk-test", secret_key="sk-test")
        result = create_langfuse_handler(settings)
        assert result is None

    def test_returns_none_with_valid_keys(self):
        settings = _Settings(
            public_key="pk-abc",
            secret_key="sk-xyz",
            host="http://custom:3000",
        )
        result = create_langfuse_handler(settings)
        assert result is None

    def test_each_call_returns_none(self):
        """The stub consistently returns None for every call."""
        settings = _Settings(public_key="pk-test", secret_key="sk-test")
        h1 = create_langfuse_handler(settings)
        h2 = create_langfuse_handler(settings)
        assert h1 is None
        assert h2 is None

    def test_returns_none_on_unexpected_exception(self):
        """Import or init errors should not propagate — tracing is optional."""
        settings = _Settings(public_key="pk-test")
        with patch("langfuse.Langfuse", side_effect=RuntimeError("boom")):
            result = create_langfuse_handler(settings)

        assert result is None
