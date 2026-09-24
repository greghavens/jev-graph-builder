"""Runtime configuration: `jev-graph-builder.yaml` + environment (pydantic-settings).

Only *where things are* and *which profiles are active* live here. Every tunable
number lives in the Registry (`policies.yaml`, `profiles.yaml`), per P1.
Secrets are read from the environment only (§17).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

CONFIG_FILE_ENV = "JEV_GRAPH_BUILDER_CONFIG"
DEFAULT_CONFIG_FILE = "jev-graph-builder.yaml"

HarnessChoice = Literal["claude_code", "codex", "auto"]


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if path.is_file():
            self._data = yaml.safe_load(path.read_text()) or {}

    def get_field_value(self, field, field_name):  # pragma: no cover - required by ABC
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


class ActiveProfiles(BaseSettings):
    jev: str = "typesafe"
    embedding: str = "local"
    harness_default: str | None = None
    postgres: str = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JGB_", env_nested_delimiter="__", extra="ignore")

    dsn: str = "postgresql://localhost/jev_graph_builder"
    corpus: str = "default"
    registry_path: Path = Path("registry")
    workspace_root: Path = Path(".jgb/workspaces")
    harness: HarnessChoice = "auto"
    # Set when `dsn` points at the container `build` manages (profiles.postgres.<name>).
    local_postgres: str | None = None
    profiles: ActiveProfiles = Field(default_factory=ActiveProfiles)
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def dsn_given(self) -> bool:
        """True when a DSN was set explicitly (argument, environment or config file), not left at the default."""
        return "dsn" in self.model_fields_set

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        path = Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_FILE))
        return (init_settings, env_settings, _YamlSource(settings_cls, path), file_secret_settings)

    def resolved_registry(self) -> Path:
        return self.registry_path.resolve()


def secret(env_var: str) -> str | None:
    """Secrets come only from the environment; empty values count as missing."""
    value = os.environ.get(env_var, "").strip()
    return value or None


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_FILE))


def write_config(updates: dict[str, Any]) -> Path:
    """Merge `updates` into the config file (never secrets: those stay in the environment)."""
    path = config_path()
    doc = (yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None) or {}
    path.write_text(yaml.safe_dump({**doc, **updates}, sort_keys=False), encoding="utf-8")
    return path
