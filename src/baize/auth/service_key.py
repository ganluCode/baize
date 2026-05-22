"""Service Key authenticator.

Authenticates requests using a static service API key compared against the
SERVICE_API_KEY environment variable. Intended for internal service-to-service
calls that do not carry a user context.
"""

import logging
import os

from baize.auth.schemas import AuthResult

logger = logging.getLogger(__name__)


def authenticate_service_key(authorization: str | None) -> AuthResult | None:
    """Authenticate a request using a Bearer service API key.

    Extracts the token from the ``Authorization: Bearer <token>`` header and
    compares it against the ``SERVICE_API_KEY`` environment variable using a
    constant-time comparison.

    Args:
        authorization: The raw value of the Authorization header, or None.

    Returns:
        :class:`AuthResult` with ``user_id=None`` and ``auth_type='service'``
        if the token matches, otherwise ``None``.  Never raises an exception.
    """
    expected = os.environ.get("SERVICE_API_KEY")
    if not expected:
        return None

    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization[len("Bearer "):]
    if token != expected:
        return None

    logger.debug("Service key authentication successful")
    return AuthResult(user_id=None, auth_type="service")
