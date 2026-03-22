"""Unit tests for baize.core.observability.create_langfuse_handler()."""

from unittest.mock import MagicMock, patch

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
    """When LangFuse is configured, the factory must return a CallbackHandler."""

    def test_returns_callback_handler_instance(self):
        from langfuse.langchain import CallbackHandler

        settings = _Settings(public_key="pk-test", secret_key="sk-test")
        with patch("langfuse.Langfuse"):
            result = create_langfuse_handler(settings)

        assert result is not None
        assert isinstance(result, CallbackHandler)

    def test_langfuse_initialized_with_settings_credentials(self):
        settings = _Settings(
            public_key="pk-abc",
            secret_key="sk-xyz",
            host="http://custom:3000",
        )
        with patch("langfuse.Langfuse") as mock_lf:
            create_langfuse_handler(settings)

        mock_lf.assert_called_once_with(
            public_key="pk-abc",
            secret_key="sk-xyz",
            host="http://custom:3000",
        )

    def test_each_call_returns_new_instance(self):
        """Each request gets its own handler to avoid trace mixing."""
        settings = _Settings(public_key="pk-test", secret_key="sk-test")
        with patch("langfuse.Langfuse"):
            h1 = create_langfuse_handler(settings)
            h2 = create_langfuse_handler(settings)

        assert h1 is not h2

    def test_returns_none_on_unexpected_exception(self):
        """Import or init errors should not propagate — tracing is optional."""
        settings = _Settings(public_key="pk-test")
        with patch("langfuse.Langfuse", side_effect=RuntimeError("boom")):
            result = create_langfuse_handler(settings)

        assert result is None
