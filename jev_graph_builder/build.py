"""`jev-graph-builder build`: documents in, verified graph out. The user only approves.

Each phase resumes from the ledger and `registry_versions`, so re-running the
same command after a crash or a declined gate picks up where it stopped.

1. setup     config file, schema, corpus row.
2. registry  S0: the harness drafts the Registry from the documents, Jev verifies
             it (§8.1). Gate: the human approves, and the draft is activated.
3. graph     S1..S7. Jev decides everything: if Jev says yes it is yes, if
             Jev says no it is no.
4. verify    P3 and edge-verification audits, graph shape, pending work (§19).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_graph_builder.app import runtime
from jev_graph_builder.config import Settings, write_config
from jev_graph_builder.harness.base import HarnessUsageLimit
from jev_graph_builder.pipeline.common import RUN_STOPS, Context, RunOptions, ensure_corpus, run_stage
from jev_graph_builder.registry import gate
from jev_graph_builder.registry.gate import BOOTSTRAP_KIND
from jev_graph_builder.registry.lint import lint_registry
from jev_graph_builder.registry.loader import Registry
from jev_graph_builder.registry.versioning import Proposal, get_proposal

STOPPED_AS_ASKED = "stop_after"
BOOTSTRAP_STAGE = "bootstrap"

# (gate name, what the human is asked to approve) -> approved?
Approver = Callable[[str, dict[str, Any]], bool]
Progress = Callable[[str, dict[str, Any]], None]
OpenContext = Callable[[Settings, str | None], AbstractAsyncContextManager[Context]]


STOP_REASONS = {HarnessUsageLimit: "harness_usage_limit"}


@dataclass
class BuildReport:
    phases: dict[str, Any] = field(default_factory=dict)
    stopped: dict[str, Any] | None = None  # {"reason": <code>, ...details}

    @property
    def ok(self) -> bool:
        verify = self.phases.get("verify") or {}
        return self.stopped is None and bool(verify.get("passed"))

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "stopped": self.stopped, "phases": self.phases}


class Builder:
    def __init__(self, docs: list[str], harness: str | None, approver: Approver, approved_by: str,
                 progress: Progress | None = None, open_ctx: OpenContext = runtime, stop_after: str | None = None) -> None:
        self.docs = docs
        self.stop_after = stop_after
        self.harness = harness
        self.approver = approver
        self.approved_by = approved_by
        self.progress = progress or (lambda _phase, _info: None)
        self.open_ctx = open_ctx
        self.report = BuildReport()

    async def _with(self, fn: Callable[[Context], Awaitable[Any]]) -> Any:
        """Each phase opens a fresh runtime: activation changes the Registry."""
        async with self.open_ctx(Settings(), self.harness) as ctx:
            ctx.cache["ingest_paths"] = self.docs
            return await fn(ctx)

    def _record(self, phase: str, info: Any) -> None:
        self.report.phases[phase] = info
        self.progress(phase, info if isinstance(info, dict) else {"result": info})

    async def run(self) -> BuildReport:
        steps: list[tuple[str, Callable[[Context], Awaitable[Any]]]] = [
            ("setup", self.setup), ("registry", self.registry), ("graph", self.graph), ("verify", self.verify),
        ]
        for phase, step in steps:
            try:
                info = await self._with(step)
            except RUN_STOPS as exc:
                # Outside the stage runner (setup, S0): same stop, unfinished work stays pending.
                self.report.stopped = {"reason": STOP_REASONS[type(exc)], "phase": phase, "error": str(exc)}
                break
            self._record(phase, info)
            rounds = 0
            while phase == "graph" and info.get("evolved") and not self.report.stopped and rounds < info["max_rounds"]:
                # Jev activated new relation types: the stages that depend on them re-run under them.
                rounds += 1
                info = await self._with(self.graph)
                self._record(f"graph_after_evolution_{rounds}", info)
            if self.report.stopped:
                break
        return self.report

    # ------------------------------------------------------------------ phases

    async def setup(self, ctx: Context) -> dict[str, Any]:
        await ensure_corpus(ctx)
        return {"corpus": ctx.corpus_id, "registry_version": ctx.registry_version, "docs": self.docs}

    async def registry(self, ctx: Context) -> dict[str, Any]:
        from jev_graph_builder.pipeline.s0_bootstrap import BootstrapStage

        if await _latest(ctx, BOOTSTRAP_KIND, ("active", "superseded")):
            self._maybe_stop(BOOTSTRAP_STAGE)
            return {"skipped": "bootstrap_active", "active": ctx.registry_version}
        # A no-op when the sample, its metadata and S0's Registry inputs are unchanged (the ledger item is
        # done); otherwise a fresh draft is staged and becomes the proposal shown at the gate.
        await BootstrapStage().run(ctx, self.docs)
        row = await _latest(ctx, BOOTSTRAP_KIND, (gate.PROPOSED, gate.APPROVED))
        if row is None:
            self.report.stopped = {"reason": "no_proposal", "gate": BOOTSTRAP_KIND}
            return {}
        proposal = get_proposal(ctx.reg.root, row["version"])
        shown = {**row["summary"], "lint_errors": lint_registry(Registry(proposal.path)).errors}
        result = await self._gate(ctx, BOOTSTRAP_KIND, proposal, shown)
        self._maybe_stop(BOOTSTRAP_STAGE)
        return result

    def _maybe_stop(self, stage: str) -> bool:
        if self.report.stopped is None and stage == self.stop_after:
            self.report.stopped = {"reason": STOPPED_AS_ASKED, "stage": stage}
        return self.report.stopped is not None

    async def graph(self, ctx: Context) -> dict[str, Any]:
        from jev_graph_builder.pipeline.stages import graph_stages

        reports, evolved = [], []
        for st in graph_stages():
            rep = await run_stage(ctx, st, RunOptions(), {"paths": self.docs})
            reports.append(rep.as_dict())
            evolved.extend(rep.proposals)
            self.progress(st.name, rep.as_dict())
            if rep.paused:
                self.report.stopped = {"reason": "paused", "stage": st.name, "errors": rep.errors}
                break
            if self._maybe_stop(st.name):
                break
        return {"registry_version": ctx.registry_version, "stages": reports, "evolved": evolved,
                "max_rounds": ctx.reg.policy("ontology.max_evolution_rounds")}

    async def verify(self, ctx: Context) -> dict[str, Any]:
        from jev_graph_builder.audit.report import edge_verification_violations, graph_summary, p3_violations
        from jev_graph_builder.pipeline.stages import graph_stages
        from jev_graph_builder.planner import build_plan

        p3 = await p3_violations(ctx.reg, ctx.db, ctx.corpus_id)
        edges = await edge_verification_violations(ctx.reg, ctx.db, ctx.corpus_id)
        pending = (await build_plan(ctx, graph_stages())).totals()["runnable_items"]
        summary = await graph_summary(ctx.db, ctx.corpus_id)
        checks = {"p3_violations": p3["total"], "edge_verification_violations": len(edges), "pending_items": pending}
        passed = not any(checks.values())
        return {"passed": passed, "checks": checks, "graph": summary, "p3": p3, "edge_verification": edges}

    # -------------------------------------------------------------------- gate

    async def _gate(self, ctx: Context, kind: str, proposal: Proposal, shown: dict[str, Any]) -> dict[str, Any]:
        if shown.get("lint_errors"):
            self.report.stopped = {"reason": "lint_errors", "gate": kind, "proposal": proposal.version}
            return {"proposal": proposal.version, **shown}
        if not self.approver(kind, {"proposal": proposal.version, **shown}):
            # re-running `build` asks again
            self.report.stopped = {"reason": "not_approved", "gate": kind, "proposal": proposal.version}
            return {"proposal": proposal.version, "approved": False}
        approval = await gate.approve(ctx, proposal, self.approved_by)
        activation = await gate.activate(ctx, proposal)
        return {"proposal": proposal.version, "approval": approval, "activation": activation}


def check_stop_after(stage: str | None) -> None:
    from jev_graph_builder.pipeline.stages import graph_stages

    names = [BOOTSTRAP_STAGE, *(st.name for st in graph_stages())]
    if stage is not None and stage not in names:
        raise ValueError(f"--stop-after must be one of {', '.join(names)}")


async def _latest(ctx: Context, kind: str, statuses: tuple[str, ...]) -> dict[str, Any] | None:
    return await ctx.db.fetchone(
        "SELECT version, status, summary FROM registry_versions WHERE corpus_id = %(c)s AND status = ANY(%(s)s) "
        "AND summary->>'kind' = %(kind)s "
        "ORDER BY coalesce(activated_at, approved_at, created_at) DESC LIMIT 1",
        {"c": ctx.corpus_id, "s": list(statuses), "kind": kind})


def configure(docs: list[str], corpus: str | None, db: str | None, registry: Path | None, harness: str | None) -> Settings:
    """Write what the command line gave into the config file; everything else keeps its value.

    With no DSN anywhere, start the local Postgres container (profiles.postgres) and record its DSN.
    """
    updates: dict[str, Any] = {}
    if corpus:
        updates["corpus"] = corpus
    if registry:
        updates["registry_path"] = str(registry)
    settings = Settings(**({"registry_path": registry} if registry else {}))
    if db:
        updates["local_postgres"] = None
    elif settings.local_postgres or not settings.dsn_given:
        # (Re)start the managed container every time: its host port can change after a restart.
        from jev_graph_builder.store import local_pg

        name = settings.local_postgres or settings.profiles.postgres
        db = local_pg.ensure(Registry(settings.resolved_registry()).profile("postgres", name))
        updates["local_postgres"] = name
    if db:
        updates["dsn"] = db
    if harness:
        updates["harness"] = harness
    if updates:
        write_config(updates)
    return Settings()
