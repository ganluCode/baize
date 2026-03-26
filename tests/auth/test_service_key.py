"""Unit tests for Service Key authenticator (F-004)."""


from baize.auth.schemas import AuthResult
from baize.auth.service_key import authenticate_service_key


class TestAuthenticateServiceKey:
    """Tests for the service key authenticator."""

    def test_matching_token_returns_auth_result(self, monkeypatch):
        """Token matching SERVICE_API_KEY returns AuthResult with service auth_type."""
        monkeypatch.setenv("SERVICE_API_KEY", "my-secret-service-key")
        result = authenticate_service_key("Bearer my-secret-service-key")
        assert result is not None
        assert isinstance(result, AuthResult)
        assert result.user_id is None
        assert result.auth_type == "service"

    def test_mismatched_token_returns_none(self, monkeypatch):
        """Token not matching SERVICE_API_KEY returns None without raising."""
        monkeypatch.setenv("SERVICE_API_KEY", "correct-key")
        result = authenticate_service_key("Bearer wrong-key")
        assert result is None

    def test_service_api_key_not_configured_returns_none(self, monkeypatch):
        """When SERVICE_API_KEY is not set, authenticator always returns None."""
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        result = authenticate_service_key("Bearer any-token")
        assert result is None

    def test_missing_bearer_prefix_returns_none(self, monkeypatch):
        """Authorization header without 'Bearer ' prefix returns None."""
        monkeypatch.setenv("SERVICE_API_KEY", "my-key")
        result = authenticate_service_key("my-key")
        assert result is None

    def test_none_authorization_header_returns_none(self, monkeypatch):
        """None authorization header returns None."""
        monkeypatch.setenv("SERVICE_API_KEY", "my-key")
        result = authenticate_service_key(None)
        assert result is None

    def test_exact_match_required(self, monkeypatch):
        """Partial matches are rejected."""
        monkeypatch.setenv("SERVICE_API_KEY", "my-key")
        result = authenticate_service_key("Bearer my-key-extra")
        assert result is None
