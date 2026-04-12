"""Unit tests for baize.user.auth_service."""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.user.auth_service import AuthError, AuthService  # noqa: E402
from baize.user.models import UserModel  # noqa: E402


def _make_user(*, is_active: bool = True, role: str = "user") -> UserModel:
    user = MagicMock(spec=UserModel)
    user.id = uuid.uuid4()
    user.email = "ganlu@example.com"
    user.name = "ganlu"
    user.password = "$2b$12$hashedpassword"
    user.role = role
    user.is_active = is_active
    return user


@pytest.fixture
def repo():
    return AsyncMock()


@pytest.fixture
def jwt_svc():
    return AsyncMock()


@pytest.fixture
def auth_service(repo, jwt_svc):
    return AuthService(user_repo=repo, jwt_service=jwt_svc)


@pytest.mark.asyncio
async def test_login_with_email_queries_by_email(auth_service, repo, jwt_svc):
    user = _make_user()
    repo.get_by_email.return_value = user
    jwt_svc.create_token.return_value = "token123"

    with patch("baize.user.auth_service.bcrypt.checkpw", return_value=True):
        result = await auth_service.login(login="ganlu@example.com", password="secret")

    repo.get_by_email.assert_awaited_once_with("ganlu@example.com")
    repo.get_by_name.assert_not_awaited()
    assert result.access_token == "token123"


@pytest.mark.asyncio
async def test_login_without_at_queries_by_name(auth_service, repo, jwt_svc):
    user = _make_user()
    repo.get_by_name.return_value = user
    jwt_svc.create_token.return_value = "token456"

    with patch("baize.user.auth_service.bcrypt.checkpw", return_value=True):
        result = await auth_service.login(login="ganlu", password="secret")

    repo.get_by_name.assert_awaited_once_with("ganlu")
    repo.get_by_email.assert_not_awaited()
    assert result.access_token == "token456"


@pytest.mark.asyncio
async def test_login_user_not_found_raises_401(auth_service, repo):
    repo.get_by_email.return_value = None

    with pytest.raises(AuthError) as exc_info:
        await auth_service.login(login="notexist@example.com", password="secret")

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_login_wrong_password_raises_401(auth_service, repo):
    user = _make_user()
    repo.get_by_email.return_value = user

    with patch("baize.user.auth_service.bcrypt.checkpw", return_value=False):
        with pytest.raises(AuthError) as exc_info:
            await auth_service.login(login="ganlu@example.com", password="wrong")

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_login_inactive_user_raises_403(auth_service, repo):
    user = _make_user(is_active=False)
    repo.get_by_email.return_value = user

    with patch("baize.user.auth_service.bcrypt.checkpw", return_value=True):
        with pytest.raises(AuthError) as exc_info:
            await auth_service.login(login="ganlu@example.com", password="secret")

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_login_success_response_fields(auth_service, repo, jwt_svc):
    user = _make_user()
    repo.get_by_email.return_value = user
    jwt_svc.create_token.return_value = "mytoken"

    with patch("baize.user.auth_service.bcrypt.checkpw", return_value=True):
        result = await auth_service.login(login="ganlu@example.com", password="secret")

    assert result.access_token == "mytoken"
    assert result.token_type == "bearer"
    assert result.expires_in == 86400


@pytest.mark.asyncio
async def test_logout_revokes_jwt(auth_service, jwt_svc):
    user_id = uuid.uuid4()
    jti = str(uuid.uuid4())
    jwt_svc.verify_token.return_value = {"user_id": str(user_id), "jti": jti}

    await auth_service.logout(token="sometoken")

    jwt_svc.verify_token.assert_awaited_once_with("sometoken")
    jwt_svc.revoke_token.assert_awaited_once_with(user_id=user_id, jti=jti)


@pytest.mark.asyncio
async def test_refresh_issues_new_token_and_revokes_old(auth_service, jwt_svc):
    user_id = uuid.uuid4()
    old_jti = str(uuid.uuid4())
    jwt_svc.verify_token.return_value = {"user_id": str(user_id), "role": "user", "jti": old_jti}
    jwt_svc.create_token.return_value = "newtoken"

    result = await auth_service.refresh(token="oldtoken")

    jwt_svc.verify_token.assert_awaited_once_with("oldtoken")
    jwt_svc.create_token.assert_awaited_once_with(user_id=user_id, role="user")
    jwt_svc.revoke_token.assert_awaited_once_with(user_id=user_id, jti=old_jti)
    assert result.access_token == "newtoken"
    assert result.token_type == "bearer"
