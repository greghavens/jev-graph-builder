"""Subprocess entry point for the crash/resume trials (§19): `run --all` with the fakes.

Usage: python -m tests.integration.driver DSN REGISTRY WORKSPACE DOCS LOG

Every Jev request and harness run is appended to LOG as it is *sent*, so the test
can tell which work a killed process had already paid for.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx2

from jev_graph_builder.app import Runtime
from jev_graph_builder.config import ActiveProfiles, Settings
from jev_graph_builder.ids import canonical_json
from jev_graph_builder.pipeline.common import RunOptions, run_stage
from jev_graph_builder.pipeline.stages import all_stages
from jev_graph_builder.registry.loader import Registry
from tests.integration.fakes import FakeEmbedder, FakeHarness, FakeJev


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class LoggingJev(FakeJev):
    def __init__(self, reg: Registry, log: Path) -> None:
        super().__init__(reg)
        self.log = log

    def transport(self) -> httpx2.MockTransport:
        handle = super().transport().handler

        def logged(request: httpx2.Request) -> httpx2.Response:
            with self.log.open("a") as f:
                f.write(json.dumps({"kind": "jev", "key": _digest(json.loads(request.content))}) + "\n")
            return handle(request)

        return httpx2.MockTransport(logged)


class LoggingHarness(FakeHarness):
    def __init__(self, reg: Registry, log: Path) -> None:
        super().__init__(reg)
        self.log = log

    async def run(self, spec):  # noqa: ANN001, ANN201
        inputs = ""
        if "input_file" in spec.variables:
            inputs = (spec.workspace / spec.variables["input_file"]).read_text()
        key = _digest({"prompt": spec.prompt_ref, "variables": {k: v for k, v in spec.variables.items() if k not in ("input_file", "output_file")},
                       "inputs": inputs})
        with self.log.open("a") as f:
            f.write(json.dumps({"kind": "harness", "key": key}) + "\n")
        return await super().run(spec)


async def main(dsn: str, registry: str, workspace: str, docs: str, log: str) -> int:
    settings = Settings(
        dsn=dsn, corpus="test", registry_path=Path(registry), workspace_root=Path(workspace),
        harness="claude_code", log_json=True, log_level="WARNING",
        profiles=ActiveProfiles(jev="typesafe", embedding="local"),
    )
    reg = Registry(settings.resolved_registry())
    harness = LoggingHarness(reg, Path(log))
    emb = reg.profile("embedding", settings.profiles.embedding)
    rt = Runtime(settings, jev_transport=LoggingJev(reg, Path(log)).transport(),
                 adapters={"claude_code": harness, "codex": harness}, embedder_factory=lambda: FakeEmbedder(emb))
    ctx = await rt.open()
    ctx.cache["ingest_paths"] = [docs]
    try:
        for stage in all_stages():
            rep = await run_stage(ctx, stage, RunOptions())
            if rep.errors:
                print(json.dumps({"stage": stage.name, "errors": rep.errors[:3]}), file=sys.stderr)
                return 1
    finally:
        await rt.close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TYPESAFE_API_KEY", "test-key")
    sys.exit(asyncio.run(main(*sys.argv[1:6])))
