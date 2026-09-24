"""§19 acceptance item 2: kill `run --all` at random points and resume.

A reference run goes through uninterrupted. Each trial SIGKILLs the driver
subprocess once it has sent a random number of Jev / harness requests, then
re-runs it to completion. The final state must equal the reference apart from
timestamps and run IDs, `plan` must report zero pending work, and the only
repeated paid requests are ones that could have been in flight at the kill
(plus the per-run drift sample, which bypasses the cache by design, §8.8).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from jev_graph_builder.app import Runtime
from jev_graph_builder.config import ActiveProfiles, Settings
from jev_graph_builder.planner import build_plan
from jev_graph_builder.pipeline.stages import all_stages
from jev_graph_builder.registry.loader import Registry
from tests.integration.fakes import FakeEmbedder, FakeHarness, FakeJev, write_docs
from tests.integration.snapshot import diff, snapshot

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
TRIALS = 5
SEED = 20260923
RUN_TIMEOUT_S = 600


def _db(server_dsn: str) -> str:
    name = "jgb_" + uuid.uuid4().hex[:12]
    with psycopg.connect(server_dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    return server_dsn.rpartition("/")[0] + "/" + name


def _drop(server_dsn: str, dsn: str) -> None:
    with psycopg.connect(server_dsn, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{dsn.rpartition("/")[2]}" WITH (FORCE)')


class Trial:
    def __init__(self, server_dsn: str, base: Path, docs: Path, registry: Path) -> None:
        self.dsn = _db(server_dsn)
        self.base = base
        self.registry = base / "registry"
        shutil.copytree(registry, self.registry)
        self.workspace = base / "ws"
        self.docs = docs
        self.log = base / "requests.jsonl"
        self.log.touch()

    def sent(self) -> int:
        return sum(1 for _ in self.log.open())

    def start(self) -> subprocess.Popen[bytes]:
        env = {**os.environ, "TYPESAFE_API_KEY": "test-key", "PYTHONPATH": str(ROOT)}
        return subprocess.Popen(
            [sys.executable, "-m", "tests.integration.driver", self.dsn, str(self.registry), str(self.workspace),
             str(self.docs), str(self.log)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=(self.base / "stderr.log").open("ab"))

    def run_to_end(self) -> None:
        proc = self.start()
        assert proc.wait(timeout=RUN_TIMEOUT_S) == 0, (self.base / "stderr.log").read_text()[-2000:]

    def run_and_kill(self, after: int) -> bool:
        """SIGKILL once `after` requests have been sent. False if the run finished first."""
        proc = self.start()
        deadline = time.monotonic() + RUN_TIMEOUT_S
        while proc.poll() is None and time.monotonic() < deadline:
            if self.sent() >= after:
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                return True
            time.sleep(0.02)
        assert proc.poll() == 0, (self.base / "stderr.log").read_text()[-2000:]
        return False

    def counts(self) -> tuple[Counter[str], Counter[str]]:
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        return (Counter(r["key"] for r in rows if r["kind"] == "jev"),
                Counter(r["key"] for r in rows if r["kind"] == "harness"))


def _excess(trial: Counter[str], reference: Counter[str]) -> int:
    return sum(max(n - reference.get(k, 0), 0) for k, n in trial.items())


async def _pending(dsn: str, registry: Path, workspace: Path) -> list[dict]:
    settings = Settings(dsn=dsn, corpus="test", registry_path=registry, workspace_root=workspace,
                        harness="claude_code", log_json=False, log_level="WARNING",
                        profiles=ActiveProfiles(jev="typesafe", embedding="local"))
    reg = Registry(registry)
    fake = FakeHarness(reg)
    emb = reg.profile("embedding", "local")
    rt = Runtime(settings, jev_transport=FakeJev(reg).transport(), adapters={"claude_code": fake, "codex": fake},
                 embedder_factory=lambda: FakeEmbedder(emb))
    ctx = await rt.open()
    try:
        plan = await build_plan(ctx, all_stages())
        return [s.as_dict() for s in plan.stages if s.runnable]
    finally:
        await rt.close()


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    write_docs(tmp_path / "docs", ROOT / "tests/fixtures/corpus")
    return tmp_path / "docs"


@pytest.fixture()
def registry_copy(tmp_path: Path) -> Path:
    """One copy for every trial, so edits to the source tree during the run cannot split versions."""
    dest = tmp_path / "registry"
    shutil.copytree(ROOT / "registry", dest, ignore=shutil.ignore_patterns(".proposals"))
    return dest


@pytest.fixture()
def trials(server_dsn: str, tmp_path: Path, docs_dir: Path) -> Iterator[list[Trial]]:
    made: list[Trial] = []
    yield made
    for t in made:
        _drop(server_dsn, t.dsn)


async def test_kill_and_resume(server_dsn: str, tmp_path: Path, docs_dir: Path, registry_copy: Path, trials: list[Trial],
                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    ref = Trial(server_dsn, tmp_path / "reference", docs_dir, registry_copy)
    trials.append(ref)
    ref.run_to_end()
    ref_state = snapshot(ref.dsn)
    ref_jev, ref_harness = ref.counts()
    total = ref.sent()
    assert total > TRIALS

    pol = Registry(ref.registry)
    in_flight = max(pol.policy("run.concurrency").values())
    drift_sample = pol.policy("audit.drift.sample_size")

    rng = random.Random(SEED)
    for i in range(TRIALS):
        t = Trial(server_dsn, tmp_path / f"trial{i}", docs_dir, registry_copy)
        trials.append(t)
        kill_at = rng.randrange(1, total)
        killed = t.run_and_kill(kill_at)
        t.run_to_end()
        processes = 1 + int(killed)

        state = snapshot(t.dsn)
        if state != ref_state:
            # pytest truncates the assertion message; keep the whole diff next to the trial.
            (t.base / "diff.json").write_text(json.dumps(diff(ref_state, state, limit=50), indent=1))
        assert state == ref_state, (i, kill_at, t.base / "diff.json")
        assert await _pending(t.dsn, t.registry, t.workspace) == [], (i, kill_at)
        jev, harness = t.counts()
        # Repeats come only from requests in flight at the kill, plus drift samples: the sample is keyed
        # by run_id (§8.8), so every process that reaches S7 may re-ask a different set than the reference.
        assert _excess(jev, ref_jev) <= int(killed) * in_flight + processes * drift_sample, (i, kill_at)
        assert _excess(harness, ref_harness) <= int(killed) * in_flight, (i, kill_at)
