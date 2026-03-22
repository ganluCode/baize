"""Authentication result schema shared across all authenticators."""

from dataclasses import dataclass


@dataclass
class AuthResult:
    """Result returned by a successful authentication attempt.

    Attributes:
        user_id: The authenticated user's ID, or None for service-level auth.
        auth_type: One of 'service', 'jwt', or 'api_key'.
    """

    user_id: str | None
    auth_type: str
