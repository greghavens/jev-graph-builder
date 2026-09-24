"""A local Postgres + pgvector for `build` when no DSN was given (§4.3: "Docker locally").

The container and its data volume are reused across runs, so re-running `build`
resumes against the same database. It listens on 127.0.0.1 only and uses trust
auth, so there is no password to store. Everything about it comes from
`profiles.postgres.<name>`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

POLL_S = 1.0


class LocalPostgresError(Exception):
    pass


def _runtime(profile: dict[str, Any]) -> str:
    for name in profile["runtimes"]:
        if shutil.which(name):
            return name
    raise LocalPostgresError(f"no container runtime found (tried {profile['runtimes']}); pass --db <dsn> instead")


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(cmd, capture_output=True, text=True)
    if check and done.returncode:
        raise LocalPostgresError(f"{' '.join(cmd[:3])} failed: {done.stderr.strip()}")
    return done


def _create(rt: str, name: str, profile: dict[str, Any]) -> None:
    # Host port chosen by the runtime (a fixed one can already be taken); read back by `ensure`.
    env = [arg for key, value in profile["env"].items() for arg in ("-e", f"{key}={value}")]
    run = _run([rt, "run", "-d", "--name", name, "-v", f"{profile['volume']}:{profile['data_dir']}",
                "-p", "127.0.0.1::5432", *env, profile["image"]], check=False)
    if run.returncode:
        _run([rt, "rm", "-f", name], check=False)  # a half-created container would block the next try
        raise LocalPostgresError(f"{rt} run failed: {run.stderr.strip()}")


def ensure(profile: dict[str, Any]) -> str:
    """Start (or reuse) the container, wait until it accepts TCP connections, return its DSN."""
    rt, name = _runtime(profile), profile["container"]
    state = _run([rt, "inspect", "--format", "{{.State.Running}}", name], check=False)
    if state.returncode:
        _create(rt, name, profile)
    elif state.stdout.strip() != "true" and _run([rt, "start", name], check=False).returncode:
        # A stopped container can fail to restart (its old host port was taken meanwhile, or the runtime
        # kept stale state). The data lives in the named volume, so recreating the container loses nothing.
        _run([rt, "rm", "-f", name], check=False)
        _create(rt, name, profile)
    # Over TCP, not the socket: during first-time init the server listens on the socket only.
    probe = [rt, "exec", name, "pg_isready", "-h", "127.0.0.1", "-U", profile["user"], "-d", profile["database"]]
    deadline = time.monotonic() + float(profile["ready_timeout_s"])
    while _run(probe, check=False).returncode:
        if time.monotonic() > deadline:
            logs = _run([rt, "logs", "--tail", "20", name], check=False)
            raise LocalPostgresError(f"{name} not ready after {profile['ready_timeout_s']}s:\n{logs.stdout}{logs.stderr}")
        time.sleep(POLL_S)
    port = _run([rt, "port", name, "5432"]).stdout.split()[0].rsplit(":", 1)[1]
    return f"postgresql://{profile['user']}@127.0.0.1:{port}/{profile['database']}"
