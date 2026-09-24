"""local_pg.ensure: a stopped container that will not restart is recreated on its volume."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from jev_graph_builder.store import local_pg

PROFILE: dict[str, Any] = {
    "runtimes": ["podman"], "container": "pg", "volume": "pgdata", "data_dir": "/var/lib/postgresql/data",
    "image": "pgvector/pgvector:pg16", "env": {"POSTGRES_HOST_AUTH_METHOD": "trust"}, "user": "postgres",
    "database": "postgres", "ready_timeout_s": 5,
}


def _fake(start_rc: int) -> tuple[list[list[str]], Any]:
    calls: list[list[str]] = []

    def run(cmd: list[str], capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        verb = cmd[1]
        rc, out = 0, ""
        if verb == "inspect":
            out = "false"
        elif verb == "start":
            rc = start_rc
        elif verb == "port":
            out = "127.0.0.1:40001\n"
        return subprocess.CompletedProcess(cmd, rc, out, "stale runtime state" if rc else "")

    return calls, run


@pytest.mark.parametrize("start_rc, recreated", [(0, False), (1, True)])
def test_restart_or_recreate(monkeypatch: pytest.MonkeyPatch, start_rc: int, recreated: bool) -> None:
    calls, run = _fake(start_rc)
    monkeypatch.setattr(local_pg.shutil, "which", lambda _n: "/usr/bin/podman")
    monkeypatch.setattr(local_pg.subprocess, "run", run)

    assert local_pg.ensure(PROFILE) == "postgresql://postgres@127.0.0.1:40001/postgres"
    verbs = [c[1] for c in calls]
    assert ("rm" in verbs and "run" in verbs) == recreated, verbs
    if recreated:
        create = next(c for c in calls if c[1] == "run")
        assert "pgdata:/var/lib/postgresql/data" in create, "recreated on the same data volume"
