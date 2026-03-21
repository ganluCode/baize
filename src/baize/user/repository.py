"""User repository for database access operations."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from baize.user.models import UserModel


class UserRepository:
    """Data access layer for UserModel."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: uuid.UUID) -> UserModel | None:
        """Return the user with the given id, or None if not found."""
        result = await self._session.execute(select(UserModel).where(UserModel.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> UserModel | None:
        """Return the user with the given email, or None if not found."""
        result = await self._session.execute(select(UserModel).where(UserModel.email == email))
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> UserModel | None:
        """Return the user with the given name, or None if not found."""
        result = await self._session.execute(select(UserModel).where(UserModel.name == name))
        return result.scalar_one_or_none()

    async def get_by_api_key_hash(self, api_key_hash: str) -> UserModel | None:
        """Return the user with the given API key hash, or None if not found."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.api_key_hash == api_key_hash)
        )
        return result.scalar_one_or_none()

    async def create(self, data: dict) -> UserModel:
        """Create a new user from data dict and return the persisted UserModel."""
        user = UserModel(**data)
        self._session.add(user)
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def update(self, user: UserModel, data: dict) -> UserModel:
        """Update only the fields present in data and persist the changes."""
        for key, value in data.items():
            setattr(user, key, value)
        user.updated_at = datetime.now(tz=timezone.utc)
        await self._session.commit()
        await self._session.refresh(user)
        return user
