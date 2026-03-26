"""Unit tests for baize.user.service (UserService)."""

import os
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from baize.agent.service import AgentConfigService  # noqa: E402
from baize.user.models import UserModel  # noqa: E402
from baize.user.schemas import UserCreateRequest, UserUpdateRequest  # noqa: E402
from baize.user.service import UserService, UserServiceError  # noqa: E402

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _make_user(
    *,
    user_id: uuid.UUID | None = None,
    name: str = "ganlu",
    email: str = "ganlu@example.com",
    role: str = "user",
    is_active: bool = True,
    preferences: dict | None = None,
    api_key_hash: str | None = "somehash",
) -> UserModel:
    """Build a mock UserModel with all fields set. Avoids passing name= to MagicMock()."""
    user = MagicMock(spec=UserModel)
    user.id = user_id or uuid.uuid4()
    user.email = email
    user.name = name
    user.password = "$2b$12$hashedpassword"
    user.role = role
    user.avatar = None
    user.is_active = is_active
    user.preferences = preferences
    user.api_key_hash = api_key_hash
    user.created_at = _NOW
    user.updated_at = _NOW
    return user


@pytest.fixture
def repo():
    mock = AsyncMock()
    mock.get_by_name = AsyncMock(return_value=None)
    mock.get_by_id = AsyncMock(return_value=None)
    mock.list_with_total = AsyncMock(return_value=([], 0))
    return mock


@pytest.fixture
def svc(repo):
    return UserService(user_repo=repo)


# ---------------------------------------------------------------------------
# get_me
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_me_returns_user_response(svc):
    user = _make_user()
    result = await svc.get_me(user)
    assert result.id == user.id
    assert result.email == user.email
    assert result.name == user.name
    # UserResponse must NOT expose password or api_key_hash
    assert not hasattr(result, "password")
    assert not hasattr(result, "api_key_hash")


# ---------------------------------------------------------------------------
# update_me
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_me_merges_preferences(svc, repo):
    user = _make_user(preferences={"language": "zh", "communication_style": "formal"})
    updated = _make_user(preferences={"language": "en", "communication_style": "formal"})
    repo.update.return_value = updated

    update_req = UserUpdateRequest(preferences={"language": "en"})
    await svc.update_me(user, update_req)

    call_data = repo.update.call_args[0][1]
    assert call_data["preferences"]["language"] == "en"
    assert call_data["preferences"]["communication_style"] == "formal"


@pytest.mark.asyncio
async def test_update_me_updates_name(svc, repo):
    user = _make_user(name="ganlu")
    repo.get_by_name.return_value = None
    updated = _make_user(name="newname")
    repo.update.return_value = updated

    result = await svc.update_me(user, UserUpdateRequest(name="newname"))
    assert result.name == "newname"


@pytest.mark.asyncio
async def test_update_me_name_taken_raises_409(svc, repo):
    user = _make_user(name="ganlu")
    other_user = _make_user(name="taken")
    repo.get_by_name.return_value = other_user

    with pytest.raises(UserServiceError) as exc_info:
        await svc.update_me(user, UserUpdateRequest(name="taken"))

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_update_me_same_name_does_not_raise(svc, repo):
    """Updating to the same name as the current user should not raise 409."""
    user = _make_user(name="ganlu")
    # get_by_name returns the same user (same id → not a conflict)
    repo.get_by_name.return_value = user
    repo.update.return_value = _make_user(name="ganlu")

    # Should not raise
    result = await svc.update_me(user, UserUpdateRequest(name="ganlu"))
    assert result.name == "ganlu"


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_user_returns_plaintext_api_key(svc, repo):
    created_user = _make_user(email="newuser@example.com", name="newuser")
    repo.create.return_value = created_user

    req = UserCreateRequest(email="newuser@example.com", name="newuser", password="pass123")
    result = await svc.create_user(req)

    assert hasattr(result, "api_key")
    assert len(result.api_key) >= 32
    # api_key in response must differ from what is stored in DB
    stored = repo.create.call_args[0][0]["api_key_hash"]
    assert result.api_key != stored


@pytest.mark.asyncio
async def test_create_user_stores_hash_not_plaintext(svc, repo):
    created_user = _make_user(email="u@example.com", name="uname")
    repo.create.return_value = created_user

    req = UserCreateRequest(email="u@example.com", name="uname", password="pw")
    result = await svc.create_user(req)

    stored_hash = repo.create.call_args[0][0]["api_key_hash"]
    assert stored_hash != result.api_key
    # sha256 hex digest is 64 chars
    assert len(stored_hash) == 64


@pytest.mark.asyncio
async def test_create_user_hashes_password(svc, repo):
    created_user = _make_user(email="u@example.com", name="uname")
    repo.create.return_value = created_user

    req = UserCreateRequest(email="u@example.com", name="uname", password="plainpw")
    await svc.create_user(req)

    stored_password = repo.create.call_args[0][0]["password"]
    assert stored_password != "plainpw"
    assert stored_password.startswith("$2b$")


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_users_returns_items_and_total(svc, repo):
    users = [_make_user(name=f"user{i}", email=f"user{i}@example.com") for i in range(3)]
    repo.list_with_total.return_value = (users, 3)

    result = await svc.list_users(skip=0, limit=10)

    assert result.total == 3
    assert len(result.items) == 3
    for item in result.items:
        assert not hasattr(item, "api_key_hash")


@pytest.mark.asyncio
async def test_list_users_passes_pagination_to_repo(svc, repo):
    repo.list_with_total.return_value = ([], 0)

    await svc.list_users(skip=10, limit=5)

    repo.list_with_total.assert_awaited_once_with(skip=10, limit=5)


# ---------------------------------------------------------------------------
# reset_api_key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_api_key_returns_new_plaintext_key(svc, repo):
    user = _make_user(api_key_hash="oldhash")
    repo.get_by_id.return_value = user
    repo.update.return_value = user

    result = await svc.reset_api_key(user.id)

    assert hasattr(result, "api_key")
    assert len(result.api_key) >= 32


@pytest.mark.asyncio
async def test_reset_api_key_updates_stored_hash(svc, repo):
    user = _make_user(api_key_hash="oldhash")
    repo.get_by_id.return_value = user
    repo.update.return_value = user

    result = await svc.reset_api_key(user.id)

    new_hash = repo.update.call_args[0][1]["api_key_hash"]
    assert new_hash != "oldhash"
    assert new_hash != result.api_key
    assert len(new_hash) == 64  # sha256 hex


@pytest.mark.asyncio
async def test_reset_api_key_user_not_found_raises_404(svc, repo):
    repo.get_by_id.return_value = None

    with pytest.raises(UserServiceError) as exc_info:
        await svc.reset_api_key(uuid.uuid4())

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# create_user — default agent integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_user_calls_create_default_agent_when_service_provided(repo):
    """create_user() triggers create_default_agent when agent_config_service is set."""
    created_user = _make_user(email="u@example.com", name="uname")
    repo.create.return_value = created_user

    agent_svc = AsyncMock(spec=AgentConfigService)
    svc = UserService(user_repo=repo, agent_config_service=agent_svc)

    req = UserCreateRequest(email="u@example.com", name="uname", password="pw")
    await svc.create_user(req)

    agent_svc.create_default_agent.assert_awaited_once_with(created_user.id)


@pytest.mark.asyncio
async def test_create_user_does_not_call_create_default_agent_when_service_absent(repo):
    """create_user() skips default agent creation when no agent_config_service."""
    created_user = _make_user(email="u@example.com", name="uname")
    repo.create.return_value = created_user

    svc = UserService(user_repo=repo)  # no agent_config_service

    req = UserCreateRequest(email="u@example.com", name="uname", password="pw")
    # Should complete without error
    result = await svc.create_user(req)

    assert result.email == "u@example.com"


@pytest.mark.asyncio
async def test_create_user_continues_when_create_default_agent_fails(repo):
    """create_user() does not raise if create_default_agent raises an exception."""
    created_user = _make_user(email="u@example.com", name="uname")
    repo.create.return_value = created_user

    agent_svc = AsyncMock(spec=AgentConfigService)
    agent_svc.create_default_agent.side_effect = RuntimeError("DB error")

    svc = UserService(user_repo=repo, agent_config_service=agent_svc)

    req = UserCreateRequest(email="u@example.com", name="uname", password="pw")
    result = await svc.create_user(req)

    # User creation should still succeed
    assert result.email == "u@example.com"
