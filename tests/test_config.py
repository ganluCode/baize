"""Tests for src/baize/core/config.py (F-004)."""

import textwrap

import pytest

from baize.core.config import MissingEnvVarError, Settings, load_yaml_config


class TestLoadYamlConfig:
    def test_load_yaml_with_env_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret123")
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("api_key: ${MY_API_KEY}\n")

        result = load_yaml_config(str(yaml_file))

        assert result["api_key"] == "secret123"

    def test_load_yaml_multiple_substitutions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                database:
                  host: ${DB_HOST}
                  port: ${DB_PORT}
            """)
        )

        result = load_yaml_config(str(yaml_file))

        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432  # YAML parses unquoted numbers as int

    def test_missing_env_var_raises_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: ${NONEXISTENT_VAR}\n")

        with pytest.raises((KeyError, MissingEnvVarError)) as exc_info:
            load_yaml_config(str(yaml_file))

        assert "NONEXISTENT_VAR" in str(exc_info.value)

    def test_missing_yaml_file_returns_empty(self):
        result = load_yaml_config("/nonexistent/path/config.yaml")

        assert result == {}

    def test_yaml_without_placeholders(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("name: Baize\nversion: '0.1.0'\n")

        result = load_yaml_config(str(yaml_file))

        assert result["name"] == "Baize"
        assert result["version"] == "0.1.0"

    def test_return_type_is_dict(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\n")

        result = load_yaml_config(str(yaml_file))

        assert isinstance(result, dict)


class TestSettings:
    def test_settings_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@db:5432/test")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("ADMIN_API_KEY", "test-key")
        monkeypatch.setenv("SECRET_KEY", "test-secret")

        settings = Settings()

        assert settings.database_url == "postgresql+asyncpg://user:pass@db:5432/test"
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.admin_api_key == "test-key"
        assert settings.secret_key == "test-secret"

    def test_debug_defaults_to_false(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/db")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("ADMIN_API_KEY", "key")
        monkeypatch.setenv("SECRET_KEY", "secret")
        monkeypatch.delenv("DEBUG", raising=False)

        settings = Settings()

        assert settings.debug is False

    def test_debug_can_be_enabled(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/db")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("ADMIN_API_KEY", "key")
        monkeypatch.setenv("SECRET_KEY", "secret")
        monkeypatch.setenv("DEBUG", "true")

        settings = Settings()

        assert settings.debug is True

    def test_yaml_config_is_empty_when_file_missing(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/db")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("ADMIN_API_KEY", "key")
        monkeypatch.setenv("SECRET_KEY", "secret")
        monkeypatch.setenv("CONFIG_YAML_PATH", "/nonexistent/config.yaml")

        settings = Settings()

        assert settings.yaml_config == {}
