"""`jev-graph-builder` command line (§15).

Every command prints JSON on stdout so it can be scripted; logs go to stderr.
`--harness claude_code|codex|auto` overrides the harness per command.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

import typer

from jev_graph_builder.app import Runtime, jev_lint_fn, runtime
from jev_graph_builder.config import Settings, write_config
from jev_graph_builder.pipeline.common import Context, RunOptions, ensure_corpus, run_stage

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
registry_app = typer.Typer(no_args_is_help=True, help="Registry versions: diff, lint, approve, activate.")
qs_app = typer.Typer(no_args_is_help=True, help="Question sets: draft, report (§7.3).")
audit_app = typer.Typer(no_args_is_help=True, help="Audits: drift, graph (§16).")
app.add_typer(registry_app, name="registry")
app.add_typer(qs_app, name="qs")
app.add_typer(audit_app, name="audit")

_state: dict[str, Any] = {"harness": None}

HarnessOpt = typer.Option(None, "--harness", help="claude_code | codex | auto")


def _echo(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False))


def _settings() -> Settings:
    return Settings()


def _run(fn: Callable[[Context], Awaitable[Any]], harness: str | None = None, migrate_db: bool = True) -> Any:
    async def go() -> Any:
        async with runtime(_settings(), harness or _state["harness"], migrate_db=migrate_db) as ctx:
            return await fn(ctx)

    return asyncio.run(go())


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


@app.callback()
def main(harness: Optional[str] = HarnessOpt) -> None:
    _state["harness"] = harness


# ------------------------------------------------------------------ setup


@app.command()
def init(corpus: str = typer.Option(..., "--corpus"), db: str = typer.Option(..., "--db", help="Postgres DSN"),
         registry: Path = typer.Option(Path("registry"), "--registry")) -> None:
    """Write the config file, create the schema (migrations generated from the embedding profile) and the corpus row."""
    path = write_config({"corpus": corpus, "dsn": db, "registry_path": str(registry)})

    async def go(ctx: Context) -> dict[str, Any]:
        from jev_graph_builder.store import migrate

        await ensure_corpus(ctx)
        current, latest = await migrate.status(ctx.db)
        return {"config": str(path), "corpus": corpus, "schema_version": current, "latest": latest,
                "registry_version": ctx.registry_version}

    _echo(_run(go))


@app.command()
def bootstrap(paths: list[str] = typer.Argument(None, help="Defaults to corpus.yaml doc_sources")) -> None:
    """S0: the harness drafts the Registry from a corpus sample, Jev verifies it, a proposal is staged."""
    from jev_graph_builder.pipeline.s0_bootstrap import BootstrapStage

    async def go(ctx: Context) -> Any:
        return await BootstrapStage().run(ctx, paths or ctx.reg.corpus["doc_sources"])

    _echo(_run(go))


@app.command()
def build(docs: list[str] = typer.Argument(..., help="Document files or directories"),
          corpus: Optional[str] = typer.Option(None, "--corpus", help="Corpus id (default: the config file's)"),
          db: Optional[str] = typer.Option(None, "--db", help="Postgres DSN (default: the config file's; with none, a local pgvector container is started)"),
          registry: Optional[Path] = typer.Option(None, "--registry", help="Registry directory (default: the config file's)"),
          by: Optional[str] = typer.Option(None, "--by", help="Approver recorded at the gates (default: the OS user)"),
          stop_after: Optional[str] = typer.Option(None, "--stop-after",
                                                   help="Stop after this stage (bootstrap, ingest, segment, ...); re-run to continue")) -> None:
    """Everything, in one command: S0 Registry draft, S1..S7, verification.

    The only input asked for afterwards is approval of the drafted Registry. Re-running resumes
    where it stopped.
    """
    import getpass

    from jev_graph_builder.build import STOPPED_AS_ASKED, Builder, check_stop_after, configure

    check_stop_after(stop_after)
    configure(docs, corpus, db, registry, _state["harness"])

    def approver(kind: str, shown: dict[str, Any]) -> bool:
        typer.echo(json.dumps(shown, indent=2, sort_keys=True, default=str, ensure_ascii=False), err=True)
        what = f"proposal {shown['proposal']}" if "proposal" in shown else f"results of stage {shown.get('stage')}"
        return typer.confirm(f"Approve the {kind} {what}?", default=False, err=True)

    def progress(phase: str, info: dict[str, Any]) -> None:
        typer.echo(json.dumps({"phase": phase, **info}, sort_keys=True, default=str, ensure_ascii=False), err=True)

    builder = Builder(list(docs), _state["harness"], approver, by or getpass.getuser(), progress, stop_after=stop_after)
    report = asyncio.run(builder.run())
    _echo(report.as_dict())
    if not report.ok and (report.stopped or {}).get("reason") != STOPPED_AS_ASKED:
        raise typer.Exit(code=1)


# --------------------------------------------------------------- registry


def _active_root() -> Path:
    return _settings().resolved_registry()


@registry_app.command("diff")
def registry_diff(version: str) -> None:
    from jev_graph_builder.registry.versioning import diff_trees, get_proposal

    proposal = get_proposal(_active_root(), version)
    typer.echo(diff_trees(_active_root(), proposal.path) or "(no changes)")


@registry_app.command("list")
def registry_list() -> None:
    from jev_graph_builder.registry.versioning import list_proposals

    _echo([p.version for p in list_proposals(_active_root())])


@registry_app.command("lint")
def registry_lint(version: Optional[str] = typer.Argument(None, help="A proposal; default the active Registry"),
                  jev: bool = typer.Option(True, "--jev/--no-jev", help="Also run the Jev lint (QS.lint_question)")) -> None:
    from jev_graph_builder.registry.lint import lint_registry, lint_with_jev
    from jev_graph_builder.registry.loader import Registry
    from jev_graph_builder.registry.versioning import get_proposal

    root = get_proposal(_active_root(), version).path if version else _active_root()
    reg = Registry(root)
    report = lint_registry(reg)
    if not report.errors and jev:
        report.extend(_run(lambda ctx: lint_with_jev(reg, jev_lint_fn(ctx.jev, ctx.db))))
    _echo({"registry": reg.version, "errors": report.errors, "warnings": report.warnings})
    if report.errors:
        raise typer.Exit(code=1)


@registry_app.command("approve")
def registry_approve(version: str, by: str = typer.Option(..., "--by", help="Approver (recorded)"),
                     note: Optional[str] = typer.Option(None, "--note")) -> None:
    """The human gate (§7.3 step 5, §8.1 step 6): lint must pass before approval."""
    from jev_graph_builder.registry import gate
    from jev_graph_builder.registry.versioning import get_proposal

    proposal = get_proposal(_active_root(), version)
    try:
        _echo(_run(lambda ctx: gate.approve(ctx, proposal, by, note)))
    except gate.GateError as exc:
        _fail(str(exc))


@registry_app.command("activate")
def registry_activate(version: str) -> None:
    """Activate an approved proposal; downstream items become stale through their dependency hashes (§14.3)."""
    from jev_graph_builder.registry import gate
    from jev_graph_builder.registry.versioning import get_proposal

    proposal = get_proposal(_active_root(), version)
    try:
        _echo(_run(lambda ctx: gate.activate(ctx, proposal), migrate_db=False))
    except gate.GateError as exc:
        _fail(str(exc))


# -------------------------------------------------------------- question sets


@qs_app.command("draft")
def qs_draft(name: str, decision: Optional[str] = typer.Option(None, "--decision", help="Required for a new question set")) -> None:
    from jev_graph_builder.calibrate.author import QuestionSetAuthor

    async def go(ctx: Context) -> dict[str, Any]:
        report = await QuestionSetAuthor(ctx.reg, ctx.db, ctx.jobs, ctx.corpus_id).draft(name, decision, [])
        return report.as_dict()

    out = _run(go)
    _echo(out)
    if not out["ok"]:
        raise typer.Exit(code=1)


@qs_app.command("report")
def qs_report(name: str) -> None:
    """Live accept and reject rates of a question set."""
    from jev_graph_builder.audit.report import outcome_rates

    async def go(ctx: Context) -> dict[str, Any]:
        qs = ctx.reg.question_set(name)
        return {"question_set": qs.ref, "outcomes": [r for r in await outcome_rates(ctx.db) if r["question_set"] == qs.ref]}

    _echo(_run(go))


# --------------------------------------------------------------- pipeline


def _opts(limit: int | None, sample: float | None, dry_run: bool, resume: bool, concurrency: int | None) -> RunOptions:
    return RunOptions(limit=limit, sample=sample, dry_run=dry_run, resume_only=resume, concurrency=concurrency)


LimitOpt = typer.Option(None, "--limit")
SampleOpt = typer.Option(None, "--sample", help="Deterministic fraction of items (0..1]")
DryOpt = typer.Option(False, "--dry-run")
ResumeOpt = typer.Option(False, "--resume", help="Only finish items an earlier run left unfinished")
ConcOpt = typer.Option(None, "--concurrency")


@app.command()
def ingest(paths: list[str], limit: Optional[int] = LimitOpt, sample: Optional[float] = SampleOpt,
           dry_run: bool = DryOpt, resume: bool = ResumeOpt, concurrency: Optional[int] = ConcOpt) -> None:
    """S1: parse, normalize and triage documents."""
    from jev_graph_builder.pipeline.s1_ingest import IngestStage

    async def go(ctx: Context) -> Any:
        ctx.cache["ingest_paths"] = paths
        report = await run_stage(ctx, IngestStage(), _opts(limit, sample, dry_run, resume, concurrency), {"paths": paths})
        return report.as_dict()

    _echo(_run(go))


@app.command()
def run(stage: Optional[str] = typer.Argument(None), all_stages: bool = typer.Option(False, "--all"),
        limit: Optional[int] = LimitOpt, sample: Optional[float] = SampleOpt, dry_run: bool = DryOpt,
        resume: bool = ResumeOpt, concurrency: Optional[int] = ConcOpt) -> None:
    """Run one graph stage, or S1..S7 in order (`--all`). Idempotent; resumes from the ledger."""
    from jev_graph_builder.pipeline.stages import graph_stage_by_name, graph_stages

    if not all_stages and not stage:
        _fail("give a stage name or --all")
    stages = graph_stages() if all_stages else [graph_stage_by_name(stage)]  # type: ignore[arg-type]
    opts = _opts(limit, sample, dry_run, resume, concurrency)

    async def go(ctx: Context) -> Any:
        reports = []
        for st in stages:
            rep = await run_stage(ctx, st, opts, {"limit": limit, "sample": sample})
            reports.append(rep.as_dict())
            if rep.paused:
                break
        return reports

    _echo(_run(go))


@app.command()
def plan() -> None:
    """What would run (missing / stale / failed / pending per stage) and the time estimate (§18)."""
    from jev_graph_builder.pipeline.stages import graph_stages
    from jev_graph_builder.planner import build_plan

    async def go(ctx: Context) -> Any:
        p = await build_plan(ctx, graph_stages())
        return {"stages": [s.as_dict() for s in p.stages], "totals": p.totals()}

    _echo(_run(go))


# ------------------------------------------------------------------- query


@app.command()
def search(query: str, k: Optional[int] = typer.Option(None, "-k")) -> None:
    from jev_graph_builder.query.search import search as do_search

    _echo(_run(lambda ctx: do_search(ctx, query, k), migrate_db=False))


@app.command()
def serve(port: int = typer.Option(..., "--port"), host: str = typer.Option("127.0.0.1", "--host")) -> None:
    """HTTP API (§9). Uses `JGB_READER_DSN` when set (least privilege, §17)."""
    import uvicorn

    from jev_graph_builder.query.http_api import create_app

    settings = _settings()
    reader = os.environ.get("JGB_READER_DSN")
    if reader:
        settings = settings.model_copy(update={"dsn": reader})
    runtimes: dict[int, Runtime] = {}

    async def open_ctx() -> Context:
        rt = Runtime(settings, _state["harness"])
        ctx = await rt.open(migrate_db=False)
        runtimes[id(ctx)] = rt
        return ctx

    async def close_ctx(ctx: Context) -> None:
        await runtimes.pop(id(ctx)).close()

    uvicorn.run(create_app(open_ctx, close_ctx), host=host, port=port)


# ------------------------------------------------------------------- audit


@audit_app.command("drift")
def audit_drift() -> None:
    """Re-ask a sample of stored decisions against the live model (§13.3)."""
    from jev_graph_builder.pipeline.s7_audit import AuditStage

    async def go(ctx: Context) -> Any:
        run_id = await ctx.ledger.start_run("audit_drift", {}, ctx.registry_version, "")
        ctx.set_run(run_id)
        out = await AuditStage().drift(ctx)
        await ctx.ledger.finish_run(run_id, "done", out or {})
        return out

    _echo(_run(go))


@audit_app.command("graph")
def audit_graph(limit: Optional[int] = LimitOpt) -> None:
    """P3 (every graph element carries its verifying decision) and graph shape."""
    from jev_graph_builder.audit.report import edge_verification_violations, graph_summary, p3_violations

    async def go(ctx: Context) -> Any:
        return {"summary": await graph_summary(ctx.db, ctx.corpus_id),
                "p3": await p3_violations(ctx.reg, ctx.db, ctx.corpus_id, limit),
                "edge_verification": await edge_verification_violations(ctx.reg, ctx.db, ctx.corpus_id)}

    _echo(_run(go, migrate_db=False))


@app.command()
def status() -> None:
    """Per-stage counts, accept / reject rates, alerts (§16)."""
    from jev_graph_builder.audit.report import alerts, outcome_rates, stage_counts, verification_pass_rate

    async def go(ctx: Context) -> Any:
        return {
            "registry_version": ctx.registry_version, "jev_model": ctx.jev.provider.model,
            "stages": await stage_counts(ctx.db),
            "outcomes": await outcome_rates(ctx.db),
            "harness_verification": await verification_pass_rate(ctx.reg, ctx.db),
            "alerts": await alerts(ctx.db, ctx.reg.policy("status.alerts_shown")),
        }

    _echo(_run(go, migrate_db=False))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
