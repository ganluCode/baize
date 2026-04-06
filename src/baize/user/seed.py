"""Seed admin user on first startup.

Reads ADMIN_USERNAME / ADMIN_PASSWORD / ADMIN_API_KEY from the environment.
Missing values are auto-generated and logged at INFO level.
The operation is idempotent: if system_users is non-empty it does nothing.
"""

import hashlib
import logging
import os
import secrets

import bcrypt
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from baize.core.database import AsyncSessionLocal
from baize.user.models import UserModel

logger = logging.getLogger(__name__)


async def seed_admin_user(engine: AsyncEngine | None = None) -> None:
    """Create the initial admin user if no users exist.

    Args:
        engine: Unused; kept for forward-compatibility. Session is obtained
                from the module-level AsyncSessionLocal.
    """
    async with AsyncSessionLocal() as session:
        count_result = await session.execute(
            select(func.count()).select_from(UserModel)
        )
        total = count_result.scalar_one()

        if total > 0:
            logger.info("seed_admin_user: users already exist (%d), skipping.", total)
            return

        username = os.environ.get("ADMIN_USERNAME", "admin")
        email = os.environ.get("ADMIN_EMAIL", f"{username}@localhost")

        password = os.environ.get("ADMIN_PASSWORD")
        password_generated = password is None
        if password_generated:
            password = secrets.token_urlsafe(16)

        api_key = os.environ.get("ADMIN_API_KEY")
        api_key_generated = api_key is None
        if api_key_generated:
            api_key = secrets.token_urlsafe(32)

        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        admin = UserModel(
            email=email,
            name=username,
            password=hashed_password,
            role="admin",
            api_key_hash=api_key_hash,
        )
        session.add(admin)
        await session.commit()

        logger.info("seed_admin_user: created admin user (username=%s).", username)
        if password_generated:
            logger.info("seed_admin_user: ADMIN_PASSWORD not set — generated: %s", password)
        if api_key_generated:
            logger.info("seed_admin_user: ADMIN_API_KEY not set — generated: %s", api_key)
