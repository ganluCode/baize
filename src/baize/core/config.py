"""Configuration loading for Baize.

Loads secret fields from environment variables via Pydantic Settings,
and parses config/config.yaml with ${ENV_VAR} placeholder substitution.
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from baize.memory.config import MemoryConfig

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

# Default path relative to project root; can be overridden via CONFIG_YAML_PATH env var.
_DEFAULT_CONFIG_YAML = str(Path(__file__).parents[3] / "config" / "config.yaml")


class MissingEnvVarError(KeyError):
    """Raised when a ${VAR} placeholder in config.yaml has no matching env var."""

    def __init__(self, var_name: str) -> None:
        self.var_name = var_name
        super().__init__(f"Missing environment variable referenced in config.yaml: '{var_name}'")


def _substitute_env_vars(text: str) -> str:
    """Replace all ${VAR} placeholders in *text* with environment variable values.

    Raises:
        MissingEnvVarError: If any referenced variable is absent from the environment.
    """

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name not in os.environ:
            raise MissingEnvVarError(var_name)
        return os.environ[var_name]

    return _ENV_VAR_PATTERN.sub(_replace, text)


def load_yaml_config(path: str | None = None) -> dict[str, Any]:
    """Read a YAML config file and substitute ${ENV_VAR} placeholders.

    Args:
        path: Absolute path to the YAML file. Defaults to config/config.yaml
              relative to the project root. May also be set via the
              CONFIG_YAML_PATH environment variable.

    Returns:
        Parsed YAML content with all placeholders resolved, or an empty dict
        when the file does not exist.

    Raises:
        MissingEnvVarError: If the file exists but references an undefined env var.
    """
    resolved_path = path or os.environ.get("CONFIG_YAML_PATH", _DEFAULT_CONFIG_YAML)

    if not Path(resolved_path).exists():
        logger.debug("Config YAML not found at %s — using empty config.", resolved_path)
        return {}

    raw = Path(resolved_path).read_text(encoding="utf-8")
    substituted = _substitute_env_vars(raw)
    return yaml.safe_load(substituted) or {}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config.yaml."""

    # --- secrets / env-only fields ---
    database_url: str
    redis_url: str
    admin_api_key: str
    secret_key: str

    # Optional fields
    langsmith_api_key: str = ""
    doubao_api_key: str = ""
    anthropic_api_key: str = ""
    newapi_api_key: str = ""

    server_host: str = "0.0.0.0"
    server_port: int = 8000
    debug: bool = False

    # OpenMemory
    openmemory_base_url: str | None = None
    openmemory_api_key: str | None = None

    # Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://langfuse:3000"
    log_level: str = "INFO"

    # --- yaml config (populated after model init) ---
    yaml_config: dict[str, Any] = {}

    # --- memory config (populated from yaml in model_post_init) ---
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    model_config = {"env_file": (".env", ".env.local"), "env_file_encoding": "utf-8", "extra": "ignore"}

    def model_post_init(self, __context: Any) -> None:
        """Load and attach the YAML config after field validation."""
        yaml_cfg = load_yaml_config()
        object.__setattr__(self, "yaml_config", yaml_cfg)
        yaml_memory = yaml_cfg.get("memory")
        if yaml_memory:
            object.__setattr__(self, "memory", MemoryConfig.model_validate(yaml_memory))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton :class:`Settings` instance (cached after first call)."""
    return Settings()
