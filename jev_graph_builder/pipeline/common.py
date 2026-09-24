"""Stage runner (§8, §14, §16).

Every stage is `run(stage, selector, registry_version) -> RunReport` over ledger
items. A stage supplies:

  * `enumerate(ctx)` -> WorkItems with an `input_hash` over every input that
    affects the output (upstream IDs/content, Registry artifact hashes, the
    relevant profile fields and the Jev model);
  * optionally `prepare(ctx, items)` for batched work (harness jobs, embedding
    batches, fan-out packing) whose results are cached for `process`;
  * `process(ctx, item)` -> a writer coroutine; the runner calls the writer and
    `Ledger.complete` in one transaction (§14.2);
  * optionally `finalize(ctx)` for corpus-wide steps, themselves ledger items.

The runner skips done items with an unchanged hash, reclaims expired leases,
honours `--limit/--sample/--dry-run`, and pauses (items stay pending) when Jev
reports its credits exhausted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection

from jev_graph_builder import log
from jev_graph_builder.config import Settings
from jev_graph_builder.embed.base import EmbeddingProvider
from jev_graph_builder.harness.base import HarnessUsageLimit
from jev_graph_builder.harness.jobs import JobRunner
from jev_graph_builder.ids import sha256_hex, unit_fraction
from jev_graph_builder.jev.client import JevCreditsExhausted
from jev_graph_builder.jev.gating import says_yes
from jev_graph_builder.jev.service import AskResult, Decision, JevService
from jev_graph_builder.ledger.ledger import DONE, REVIEW, Ledger, WorkItem
from jev_graph_builder.registry.loader import Registry
from jev_graph_builder.store import repo
from jev_graph_builder.store.db import Database

Writer = Callable[[AsyncConnection], Awaitable[str]]


# Conditions that stop the whole run (not one item): unfinished items are released, never failed.
RUN_STOPS = (JevCreditsExhausted, HarnessUsageLimit)


@dataclass(frozen=True)
class Deps:
    """Registry artifacts and profile fields whose change invalidates a stage (§14.3)."""

    question_sets: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    policies: tuple[str, ...] = ()
    ontology: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    embedding: bool = False
    corpus: bool = False
    structural: bool = False


@dataclass
class RunOptions:
    limit: int | None = None
    sample: float | None = None
    dry_run: bool = False
    resume_only: bool = False  # `--resume`: only finish items an earlier run left pending/failed/running
    concurrency: int | None = None


@dataclass
class RunReport:
    stage: str
    run_id: str | None
    selected: int = 0
    skipped: int = 0
    done: int = 0
    review: int = 0
    failed: int = 0
    paused: bool = False
    by_status: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)   # Registry versions Jev activated during this stage

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Context:
    settings: Settings
    reg: Registry
    db: Database
    ledger: Ledger
    jev: JevService
    jobs: JobRunner
    embedder_factory: Callable[[], EmbeddingProvider]
    corpus_id: str
    run_id: str | None = None
    cache: dict[str, Any] = field(default_factory=dict)
    _embedder: EmbeddingProvider | None = None

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = self.embedder_factory()
        return self._embedder

    @property
    def registry_version(self) -> str:
        return self.reg.version

    def set_run(self, run_id: str) -> None:
        self.run_id = run_id
        self.jev.run_id = run_id
        self.jobs.run_id = run_id

    # ---------------------------------------------------------------- hashing

    def deps_hash(self, deps: Deps) -> str:
        reg = self.reg
        parts: list[Any] = [self.jev.provider.model]
        for name in deps.question_sets:
            qs = reg.question_set(name)
            parts.append([qs.ref, qs.content_hash])
        if deps.question_sets:  # the threshold decides what Jev's answers mean (§11.5)
            parts.append(["gating", reg.policy("gating")])
        for name in deps.prompts:
            prompt = reg.prompt(name)
            parts.append([prompt.ref, prompt.content_hash])
        parts.extend([p, reg.policy(p)] for p in deps.policies)
        parts.extend([k, reg.ontology(k)] for k in deps.ontology)
        parts.extend([s, reg.artifact_hash(f"schemas/{s}.json")] for s in deps.schemas)
        if deps.embedding:
            prof = reg.profile("embedding", self.settings.profiles.embedding)
            parts.append([prof["model_id"], prof["dim"], prof["max_tokens"]])
        if deps.corpus:
            parts.append(reg.corpus)
        if deps.structural:
            parts.append(reg.structural_labels)
        return sha256_hex(*parts)

    # ---------------------------------------------------------------- helpers

    def flag(self, result: AskResult, question: str, decision: Decision | None = None) -> bool:
        """Jev's yes on one Noul (e.g. an injection or boilerplate flag), at the decision's threshold."""
        return self.flag_decision(decision or result.single, question)

    def flag_decision(self, decision: Decision, question: str) -> bool:
        return says_yes(decision.answers[question], decision.threshold_for(question))


def as_text(value: Any) -> Any:
    """Metadata values as text: Jev state never carries numbers to compare (§11.4)."""
    if isinstance(value, list):
        return [as_text(v) for v in value]
    if isinstance(value, dict):
        return {k: as_text(v) for k, v in value.items()}
    return value if isinstance(value, str) else str(value)


def chunk_metadata(reg: Registry, meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """The document metadata that `policies.ingest.metadata_fields` carries onto the document's chunks."""
    meta = meta or {}
    picked = {k: as_text(meta[k]) for k in reg.policy("ingest.metadata_fields") if meta.get(k) not in (None, "", [])}
    return picked or None


async def ensure_corpus(ctx: Context) -> None:
    async with ctx.db.tx() as conn:
        await repo.upsert(
            conn, "corpora",
            {"corpus_id": ctx.corpus_id, "name": ctx.reg.corpus["name"], "registry_version": ctx.registry_version},
            key=("corpus_id",), update=("registry_version",),
        )


async def write_decisions(conn: AsyncConnection, *results: AskResult | Decision | None) -> list[str]:
    """Insert decision rows; returns their IDs for provenance columns."""
    ids: list[str] = []
    for r in results:
        if r is None:
            continue
        for d in (r.decisions if isinstance(r, AskResult) else [r]):
            await repo.upsert(conn, "decisions", d.row(), key=("decision_id",))
            ids.append(d.decision_id)
    return ids


async def enqueue_review(
    conn: AsyncConnection, ctx: Context, subject_kind: str, subject_id: str, reason: str,
    decision: Decision | None = None, payload: dict[str, Any] | None = None,
) -> str:
    review_id = sha256_hex(subject_kind, subject_id, reason, decision.decision_id if decision else "")
    await repo.upsert(
        conn, "review_queue",
        {
            "review_id": review_id, "subject_kind": subject_kind, "subject_id": subject_id, "reason": reason,
            "decision_id": decision.decision_id if decision else None, "payload": payload or {},
            "status": "open", "question_set": decision.question_set if decision else None, "run_id": ctx.run_id,
        },
        key=("review_id",), update=("payload", "run_id"),
    )
    return review_id


# -------------------------------------------------------------------- runner


class Stage:
    name: str = ""
    deps: Deps = Deps()

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        raise NotImplementedError

    async def prepare(self, ctx: Context, items: list[WorkItem], enumerated: list[WorkItem]) -> None:
        """Batch work for the selected `items`. `enumerated` is every item of the stage, done or not:
        a stage that forms batches or routes on them does so over it, so a resumed run forms the
        same batches and asks Jev the same questions as an uninterrupted one."""
        return None

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        raise NotImplementedError

    async def after_item(self, ctx: Context, report: RunReport) -> None:
        """Runs after each item is written; stages that act on accumulating results override it."""
        return None

    async def finalize(self, ctx: Context, report: RunReport, opts: RunOptions) -> None:
        return None

    async def cleanup(self, ctx: Context) -> None:
        """Runs when the stage run ends, however it ends: stop background work `prepare` started."""
        return None

    def deps_hash(self, ctx: Context) -> str:
        key = f"deps:{self.name}"
        if key not in ctx.cache:
            ctx.cache[key] = ctx.deps_hash(self.deps)
        return ctx.cache[key]

    def item(self, ctx: Context, item_id: str, *inputs: Any, payload: Any = None) -> WorkItem:
        return WorkItem(self.name, item_id, sha256_hex(self.name, self.deps_hash(ctx), *inputs), payload)


def pending_in(batches: list[list[WorkItem]], selected: list[WorkItem]) -> list[tuple[list[WorkItem], list[WorkItem]]]:
    """Each stable batch (formed over every enumerated item) paired with its selected items; batches
    with nothing selected are dropped. Route on the whole batch, run only the selected part."""
    chosen = {i.item_id for i in selected}
    pairs = [(b, [i for i in b if i.item_id in chosen]) for b in batches]
    return [(b, todo) for b, todo in pairs if todo]


RUNNABLE = {"missing", "stale", "pending", "failed", "running"}
NEW = {"missing", "stale"}


async def select_items(ctx: Context, stage: Stage, opts: RunOptions) -> tuple[list[WorkItem], dict[str, int], list[WorkItem]]:
    items = await stage.enumerate(ctx)
    existing = await ctx.ledger.existing(stage.name)
    by_status: dict[str, int] = {}
    selected: list[WorkItem] = []
    for item in items:
        status = ctx.ledger.classify(item, existing.get(item.item_id))
        if status == "failed" and existing[item.item_id]["attempts"] >= ctx.ledger.max_attempts:
            status = "failed_final"
        by_status[status] = by_status.get(status, 0) + 1
        if status not in RUNNABLE or (opts.resume_only and status in NEW):
            continue
        if opts.sample is not None and unit_fraction(stage.name, item.item_id) >= opts.sample:
            continue
        selected.append(item)
    if opts.limit is not None:
        selected = selected[: opts.limit]
    return selected, by_status, items


async def run_stage(ctx: Context, stage: Stage, opts: RunOptions, selector: dict[str, Any] | None = None) -> RunReport:
    await ensure_corpus(ctx)
    await ctx.ledger.reclaim_expired()
    selected, by_status, enumerated = await select_items(ctx, stage, opts)
    report = RunReport(stage=stage.name, run_id=None, selected=len(selected), by_status=by_status)
    report.skipped = sum(n for s, n in by_status.items() if s not in RUNNABLE)
    if opts.dry_run:
        return report

    run_id = await ctx.ledger.start_run(stage.name, selector or {}, ctx.registry_version, ctx.deps_hash(stage.deps))
    ctx.set_run(run_id)
    report.run_id = run_id
    logger = log.get().bind(run_id=run_id, stage=stage.name)
    status = "done"
    try:
        logger.info("stage_started", selected=len(selected), skipped=report.skipped)
        if selected:
            await stage.prepare(ctx, selected, enumerated)
        await _process_all(ctx, stage, selected, opts, report, logger)
        if not report.paused:
            await stage.finalize(ctx, report, opts)
    except RUN_STOPS as exc:
        report.paused = True
        report.errors.append(f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        status = "failed"
        report.errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        await stage.cleanup(ctx)
        if report.paused:
            status = "paused"
        await ctx.ledger.finish_run(run_id, status, report.as_dict())
        logger.info("stage_finished", status=status, done=report.done, review=report.review, failed=report.failed)
    return report


async def _process_all(ctx: Context, stage: Stage, items: list[WorkItem], opts: RunOptions, report: RunReport, logger: Any) -> None:
    concurrency = opts.concurrency or ctx.reg.policy(f"run.concurrency.{stage.name}")
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()

    async def one(item: WorkItem) -> None:
        async with sem:
            if stop.is_set():
                return
            try:
                await run_item(ctx, stage, item, report, logger)
            except RUN_STOPS as exc:
                if not stop.is_set():
                    stop.set()
                    report.paused = True
                    report.errors.append(f"{type(exc).__name__}: {exc}")
                return
            finished = report.done + report.review + report.failed
            if finished % every == 0:
                logger.info("stage_progress", finished=finished, selected=len(items), done=report.done,
                            review=report.review, failed=report.failed)

    every = ctx.reg.policy("run.progress_every")
    await asyncio.gather(*(one(i) for i in items))


async def run_item(ctx: Context, stage: Stage, item: WorkItem, report: RunReport, logger: Any) -> None:
    if not await ctx.ledger.claim(item, ctx.run_id or "", ctx.registry_version):
        report.skipped += 1
        return
    try:
        writer = await stage.process(ctx, item)
        async with ctx.db.tx() as conn:
            status = await writer(conn)
            await ctx.ledger.complete(conn, item, status)
        if status == REVIEW:
            report.review += 1
        else:
            report.done += 1
        await stage.after_item(ctx, report)
    except RUN_STOPS:
        await ctx.ledger.release(item)
        raise
    except Exception as exc:  # the item fails; the run continues (§14.2)
        final = await ctx.ledger.fail(item, f"{type(exc).__name__}: {exc}")
        report.failed += 1
        report.errors.append(f"{item.item_id}: {type(exc).__name__}: {exc}")
        logger.warning("item_failed", item_id=item.item_id, error=str(exc), status=final, exc_info=True)


async def run_single_item(ctx: Context, stage: Stage, item: WorkItem, writer_factory: Callable[[], Awaitable[Writer]]) -> str | None:
    """Corpus-wide finalize steps are ledger items too (so they resume and skip)."""
    existing = (await ctx.ledger.existing(stage.name)).get(item.item_id)
    if ctx.ledger.classify(item, existing) in (DONE, REVIEW):
        return None
    if not await ctx.ledger.claim(item, ctx.run_id or "", ctx.registry_version):
        return None
    try:
        writer = await writer_factory()
        async with ctx.db.tx() as conn:
            status = await writer(conn)
            await ctx.ledger.complete(conn, item, status)
        return status
    except RUN_STOPS:
        await ctx.ledger.release(item)
        raise
    except Exception as exc:
        await ctx.ledger.fail(item, f"{type(exc).__name__}: {exc}")
        raise


def done_writer(fn: Callable[[AsyncConnection], Awaitable[str | None]]) -> Writer:
    async def w(conn: AsyncConnection) -> str:
        return (await fn(conn)) or DONE

    return w
