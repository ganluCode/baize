"""Auth service: login, logout, and token refresh."""

import logging
import uuid
from dataclasses import dataclass

import bcrypt

from baize.user.jwt_service import JWTService, TokenError
from baize.user.repository import UserRepository

logger = logging.getLogger(__name__)

_TOKEN_TTL_SECONDS = 86400


class AuthError(Exception):
    """Raised when authentication or authorisation fails."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TokenResponse:
    """Returned on successful login or refresh."""

    access_token: str
    token_type: str
    expires_in: int


class AuthService:
    """Handles credential verification, JWT issuance, and revocation."""

    def __init__(self, user_repo: UserRepository, jwt_service: JWTService) -> None:
        self._repo = user_repo
        self._jwt = jwt_service

    async def login(self, login: str, password: str) -> TokenResponse:
        """Authenticate a user by email or name and return a JWT.

        Args:
            login: Either an email address (contains '@') or a username.
            password: The plaintext password to verify.

        Returns:
            TokenResponse with access_token, token_type, and expires_in.

        Raises:
            AuthError: 401 if the user does not exist or the password is wrong.
            AuthError: 403 if the user account is inactive.
        """
        if "@" in login:
            user = await self._repo.get_by_email(login)
        else:
            user = await self._repo.get_by_name(login)

        if user is None:
            raise AuthError("Invalid credentials.", status_code=401)

        password_matches = bcrypt.checkpw(password.encode(), user.password.encode())
        if not password_matches:
            raise AuthError("Invalid credentials.", status_code=401)

        if not user.is_active:
            raise AuthError("User account is inactive.", status_code=403)

        token = await self._jwt.create_token(user_id=user.id, role=user.role)
        logger.info("User %s logged in.", user.id)
        return TokenResponse(access_token=token, token_type="bearer", expires_in=_TOKEN_TTL_SECONDS)

    async def logout(self, token: str) -> None:
        """Revoke the given JWT so it can no longer be used.

        Args:
            token: The JWT string to revoke.

        Raises:
            TokenError: If the token is already invalid or expired.
        """
        payload = await self._jwt.verify_token(token)
        user_id = uuid.UUID(payload["user_id"])
        jti = payload["jti"]
        await self._jwt.revoke_token(user_id=user_id, jti=jti)
        logger.info("User %s logged out (jti=%s).", user_id, jti)

    async def refresh(self, token: str) -> TokenResponse:
        """Issue a new JWT and revoke the old one.

        Args:
            token: The current valid JWT string.

        Returns:
            TokenResponse containing the new access token.

        Raises:
            TokenError: If the provided token is invalid or expired.
        """
        payload = await self._jwt.verify_token(token)
        user_id = uuid.UUID(payload["user_id"])
        role = payload["role"]
        old_jti = payload["jti"]

        new_token = await self._jwt.create_token(user_id=user_id, role=role)
        await self._jwt.revoke_token(user_id=user_id, jti=old_jti)
        logger.info("Refreshed JWT for user %s (old_jti=%s).", user_id, old_jti)
        return TokenResponse(access_token=new_token, token_type="bearer", expires_in=_TOKEN_TTL_SECONDS)
