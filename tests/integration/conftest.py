"""Integration fixtures: a fresh Postgres database per test, the seed Registry, and fakes.

Postgres comes from `JGB_TEST_DSN` (a server where the test user may CREATE DATABASE),
or from a pgvector testcontainer when Docker/podman is reachable.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest

from jev_graph_builder.app import Runtime
from jev_graph_builder.config import ActiveProfiles, Settings
from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.registry.loader import Registry
from tests.integration.fakes import FakeEmbedder, FakeHarness, FakeJev, write_docs

ROOT = Path(__file__).resolve().parents[2]
PG_IMAGE = "pgvector/pgvector:pg16"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def server_dsn() -> Iterator[str]:
    dsn = os.environ.get("JGB_TEST_DSN")
    if dsn:
        yield dsn
        return
    try:
        from testcontainers.postgres import PostgresContainer

        with PostgresContainer(PG_IMAGE, driver=None) as pg:
            yield pg.get_connection_url()
    except Exception as exc:  # no container runtime
        pytest.skip(f"no Postgres: set JGB_TEST_DSN or run a container runtime ({exc})")


@pytest.fixture()
def dsn(server_dsn: str) -> Iterator[str]:
    name = "jgb_" + uuid.uuid4().hex[:12]
    with psycopg.connect(server_dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    base, _, _ = server_dsn.rpartition("/")
    yield f"{base}/{name}"
    if os.environ.get("JGB_TEST_KEEP_DB"):  # leave it for post-mortem inspection
        print(f"kept database {name}")
        return
    with psycopg.connect(server_dsn, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture()
def registry_dir(tmp_path: Path) -> Path:
    """A private copy, so tests may stage and activate proposals."""
    import shutil

    dst = tmp_path / "registry"
    shutil.copytree(ROOT / "registry", dst, ignore=shutil.ignore_patterns(".proposals"))
    return dst


@pytest.fixture()
def settings(dsn: str, registry_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        dsn=dsn, corpus="test", registry_path=registry_dir, workspace_root=tmp_path / "ws",
        harness="claude_code", log_json=False, log_level="WARNING",
        profiles=ActiveProfiles(jev="typesafe", embedding="local"),
    )


@pytest.fixture()
def docs(tmp_path: Path) -> list[Path]:
    return write_docs(tmp_path / "docs", ROOT / "tests/fixtures/corpus")


class Harnessed:
    def __init__(self, settings: Settings, fake_jev: FakeJev, harness: FakeHarness) -> None:
        self.settings = settings
        self.fake_jev = fake_jev
        self.harness = harness

    async def open(self, docs: list[Path] | None = None) -> tuple[Runtime, Context]:
        reg = Registry(self.settings.resolved_registry())
        emb = reg.profile("embedding", self.settings.profiles.embedding)
        rt = Runtime(self.settings, jev_transport=self.fake_jev.transport(),
                     adapters={"claude_code": self.harness, "codex": self.harness},
                     embedder_factory=lambda: FakeEmbedder(emb))
        ctx = await rt.open()
        if docs:
            ctx.cache["ingest_paths"] = [str(docs[0].parent)]
        return rt, ctx


@pytest.fixture()
def env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Harnessed:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    reg = Registry(settings.resolved_registry())
    return Harnessed(settings, FakeJev(reg), FakeHarness(reg))


@pytest.fixture()
async def ctx(env: Harnessed, docs: list[Path]) -> AsyncIterator[Context]:
    rt, c = await env.open(docs)
    try:
        yield c
    finally:
        await rt.close()
