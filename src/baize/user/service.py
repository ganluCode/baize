"""User service: business logic for user management."""

import hashlib
import logging
import uuid

import bcrypt

from baize.user.api_key import generate_api_key
from baize.user.models import UserModel
from baize.user.repository import UserRepository
from baize.user.schemas import (
    ResetKeyResponse,
    UserCreateRequest,
    UserCreateResponse,
    UserListResponse,
    UserResponse,
    UserUpdateRequest,
)

logger = logging.getLogger(__name__)


class UserServiceError(Exception):
    """Raised when a user service operation fails."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _hash_for_lookup(plaintext_key: str) -> str:
    """Compute the sha256 hex digest used to look up an API key in the DB.

    This must match the lookup logic in deps.py (get_current_user_api_key).
    """
    return hashlib.sha256(plaintext_key.encode()).hexdigest()


class UserService:
    """Handles user profile retrieval, update, creation, and API key management."""

    def __init__(self, user_repo: UserRepository) -> None:
        self._repo = user_repo

    async def get_me(self, user: UserModel) -> UserResponse:
        """Return the public profile of the given user.

        Args:
            user: The authenticated UserModel instance.

        Returns:
            UserResponse without password or api_key_hash.
        """
        return UserResponse.model_validate(user)

    async def update_me(self, user: UserModel, data: UserUpdateRequest) -> UserResponse:
        """Update the current user's profile with partial data.

        Preferences are deep-merged: only the provided keys are updated;
        existing keys not present in the request are preserved.

        Args:
            user: The authenticated UserModel to update.
            data: Fields to update.

        Returns:
            Updated UserResponse.

        Raises:
            UserServiceError: 409 if the requested name is taken by another user.
        """
        update_dict: dict = {}

        if data.name is not None and data.name != user.name:
            existing = await self._repo.get_by_name(data.name)
            if existing is not None and existing.id != user.id:
                raise UserServiceError("Name already taken.", status_code=409)
            update_dict["name"] = data.name

        if data.avatar is not None:
            update_dict["avatar"] = data.avatar

        if data.preferences is not None:
            current_prefs: dict = user.preferences or {}
            update_dict["preferences"] = {**current_prefs, **data.preferences}

        if update_dict:
            user = await self._repo.update(user, update_dict)

        return UserResponse.model_validate(user)

    async def create_user(self, data: UserCreateRequest) -> UserCreateResponse:
        """Create a new user, hashing their password and generating an API key.

        The plaintext API key is returned once in the response and never stored.

        Args:
            data: User creation request.

        Returns:
            UserCreateResponse with a one-time plaintext api_key field.
        """
        hashed_password = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()

        plaintext_key = generate_api_key()
        key_hash = _hash_for_lookup(plaintext_key)

        user_data = {
            "email": data.email,
            "name": data.name,
            "password": hashed_password,
            "role": data.role,
            "avatar": data.avatar,
            "preferences": data.preferences,
            "api_key_hash": key_hash,
        }
        user = await self._repo.create(user_data)
        logger.info("Created user %s (role=%s).", user.id, user.role)

        base = UserResponse.model_validate(user).model_dump()
        return UserCreateResponse(**base, api_key=plaintext_key)

    async def list_users(self, skip: int = 0, limit: int = 20) -> UserListResponse:
        """Return a paginated list of users.

        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            UserListResponse with items and total count.
        """
        items, total = await self._repo.list_with_total(skip=skip, limit=limit)
        return UserListResponse(
            items=[UserResponse.model_validate(u) for u in items],
            total=total,
        )

    async def reset_api_key(self, user_id: uuid.UUID) -> ResetKeyResponse:
        """Generate a new API key for the given user, invalidating the old one.

        Args:
            user_id: UUID of the user whose key should be reset.

        Returns:
            ResetKeyResponse with the new plaintext API key.

        Raises:
            UserServiceError: 404 if no user with the given id exists.
        """
        user = await self._repo.get_by_id(user_id)
        if user is None:
            raise UserServiceError("User not found.", status_code=404)

        plaintext_key = generate_api_key()
        key_hash = _hash_for_lookup(plaintext_key)
        await self._repo.update(user, {"api_key_hash": key_hash})
        logger.info("Reset API key for user %s.", user_id)
        return ResetKeyResponse(api_key=plaintext_key)
