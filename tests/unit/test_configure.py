"""`build` DSN resolution: an explicit DSN wins; otherwise the managed local container is (re)started."""

from __future__ import annotations

from pathlib import Path

import pytest

from jev_graph_builder import build
from jev_graph_builder.config import CONFIG_FILE_ENV
from jev_graph_builder.store import local_pg

REGISTRY = Path(__file__).resolve().parents[2] / "registry"


@pytest.fixture
def started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jgb.yaml"))
    monkeypatch.delenv("JGB_DSN", raising=False)
    calls: list[str] = []

    def ensure(profile: dict) -> str:
        calls.append(profile["container"])
        return f"postgresql://managed/{len(calls)}"

    monkeypatch.setattr(local_pg, "ensure", ensure)
    return calls


def test_no_dsn_starts_container_and_refreshes_each_run(started: list[str]) -> None:
    first = build.configure([], None, None, REGISTRY, None)
    second = build.configure([], None, None, None, None)
    assert len(started) == 2 and first.local_postgres == "local"
    assert second.dsn == "postgresql://managed/2"  # port may change after a restart


def test_explicit_dsn_never_starts_container(started: list[str]) -> None:
    build.configure([], None, None, REGISTRY, None)
    settings = build.configure([], None, "postgresql://u@h/db", None, None)
    again = build.configure([], None, None, None, None)
    assert len(started) == 1
    assert settings.local_postgres is None and again.dsn == "postgresql://u@h/db"


def test_dsn_given_counts_every_source(started: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime starts the managed container only when no DSN was given anywhere: a DSN passed
    to `Settings` directly (as tests and library callers do) must win over the container."""
    from jev_graph_builder.config import Settings, write_config

    assert not Settings().dsn_given
    assert Settings(dsn="postgresql://u@h/db").dsn_given
    monkeypatch.setenv("JGB_DSN", "postgresql://u@h/env")
    assert Settings().dsn_given
    monkeypatch.delenv("JGB_DSN")
    write_config({"dsn": "postgresql://u@h/file"})
    assert Settings().dsn_given
