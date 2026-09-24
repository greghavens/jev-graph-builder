"""Wiring: Settings -> Context (Registry, Postgres, Jev, harnesses, embeddings).

Used by the CLI and the HTTP API. Nothing here holds domain values: profile
and policy names come from `jev-graph-builder.yaml` and the Registry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from jev_graph_builder import log
from jev_graph_builder.config import Settings
from jev_graph_builder.embed.base import build_embedder
from jev_graph_builder.harness.claude import ClaudeCodeAdapter
from jev_graph_builder.harness.codex import CodexAdapter
from jev_graph_builder.harness.jobs import JobRunner
from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.jev.client import JevProvider, build_provider
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.jev.service import JevService, write_decisions
from jev_graph_builder.jev.tokens import TokenEstimator
from jev_graph_builder.ledger.ledger import Ledger
from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.registry.loader import Registry
from jev_graph_builder.store import migrate
from jev_graph_builder.store.db import Database

QS_HARNESS_ROUTE = "harness_route"
ROUTE_SUBJECT = "harness_job"


def migration_params(reg: Registry, settings: Settings) -> dict[str, Any]:
    prof = reg.profile("embedding", settings.profiles.embedding)
    idx = reg.policy("store.hnsw")
    return {"dim": prof["dim"], "hnsw_m": idx["m"], "hnsw_ef_construction": idx["ef_construction"]}


def build_jev_provider(reg: Registry, settings: Settings, transport: Any = None) -> JevProvider:
    return build_provider(reg.profile("jev", settings.profiles.jev), reg.policies, transport=transport)


def route_sample(records: list[Any], est: TokenEstimator, max_chars: int, max_tokens: int) -> list[str]:
    """§8.4: Jev routes on a sample, never the whole batch: records clipped so each fits the token limit,
    spread evenly across the batch, as many as fit `max_tokens` together."""
    if not records:
        return []
    clip = min(max_chars, est.chars(max_tokens))
    k = min(len(records), max(1, est.chars(max_tokens) // max(1, clip)))
    while True:
        picks = [str(records[j * len(records) // k])[:clip] for j in range(k)]
        if k == 1 or est.value(picks) <= max_tokens:
            return picks
        k -= 1


def make_router(reg: Registry, jev: JevService, db: Database | None):
    """R-100: `QS.harness_route` picks the harness profile for a job (a Choice over routable profiles)."""

    async def route(job_name: str, state: dict[str, Any]) -> str | None:
        exclude = set(state.get("exclude") or [])
        records = route_sample(state.get("records") or [], jev.provider.estimator,
                               reg.policy("harness.route_sample_chars"), reg.policy("harness.route_state_tokens"))
        prompt = reg.prompt(state.get("prompt") or job_name)
        job = {"name": job_name, "purpose": prompt.meta.get("purpose"), "task": prompt.task}
        default_name = reg.policy(f"harness.defaults.{job_name}")
        # Jev chooses among the profiles at the job's autonomy level only: a batch job needs a workspace-writing one.
        autonomy = reg.profile("harness", default_name)["autonomy"]
        profiles = {k: v["description"] for k, v in reg.profiles["harness"].items()
                    if v["autonomy"] == autonomy and v.get("routable", True) and k not in exclude}
        inputs = {"job": job, "records": records}
        result = await jev.ask(QS_HARNESS_ROUTE, inputs, ROUTE_SUBJECT, sha256_hex(job_name, inputs, profiles),
                               dynamic={"profiles": profiles})
        decision = result.single
        if db is not None:
            async with db.tx() as conn:
                await write_decisions(conn, result)
        if decision.outcome == ACCEPT:
            return decision.answers[reg.policy("harness.route_question")]["choice"]
        return None if default_name in exclude else default_name

    return route


class Runtime:
    """Owns the open resources behind a Context."""

    def __init__(self, settings: Settings, harness: str | None = None, jev_transport: Any = None,
                 adapters: dict[str, Any] | None = None, embedder_factory: Any = None) -> None:
        self.settings = settings
        self.harness = harness or settings.harness
        self.jev_transport = jev_transport
        self.adapters = adapters
        self.embedder_factory = embedder_factory
        self.db: Database | None = None
        self.ctx: Context | None = None

    async def open(self, migrate_db: bool = True) -> Context:
        s = self.settings
        log.configure(s.log_level, s.log_json)
        reg = Registry(s.resolved_registry())
        pool = reg.policy("store.pool")
        dsn = s.dsn
        if s.local_postgres or not s.dsn_given:
            # Every command, not only `build`, uses the managed container: its host port can change on restart.
            from jev_graph_builder.store import local_pg

            dsn = local_pg.ensure(reg.profile("postgres", s.local_postgres or s.profiles.postgres))
        self.db = db = Database(dsn, pool["min_size"], pool["max_size"])
        await db.open()
        if migrate_db:
            params = migration_params(reg, s)
            applied = await migrate.migrate(db, params)
            if applied:
                log.get().info("migrations_applied", migrations=applied)
            if await migrate.reconcile_embedding_dim(db, params):
                log.get().warning("embedding_dimension_changed", dim=params["dim"])
        provider = build_jev_provider(reg, s, self.jev_transport)
        jev = JevService(reg, provider, db)
        adapters = self.adapters or {"claude_code": ClaudeCodeAdapter(reg), "codex": CodexAdapter(reg)}
        jobs = JobRunner(reg, adapters, db, s.workspace_root, router=make_router(reg, jev, db), harness_override=self.harness)
        emb_profile = reg.profile("embedding", s.profiles.embedding)
        ledger_pol = reg.policy("run.ledger")
        self.ctx = Context(
            settings=s, reg=reg, db=db,
            ledger=Ledger(db, ledger_pol["lease_seconds"], ledger_pol["max_attempts"], ledger_pol["heartbeat_seconds"]),
            jev=jev, jobs=jobs,
            embedder_factory=self.embedder_factory or (lambda: build_embedder(emb_profile)),
            corpus_id=s.corpus,
        )
        return self.ctx

    async def close(self) -> None:
        if self.ctx is not None:
            await self.ctx.jev.provider.aclose()
        if self.db is not None:
            await self.db.close()


@asynccontextmanager
async def runtime(settings: Settings, harness: str | None = None, migrate_db: bool = True, **kw: Any) -> AsyncIterator[Context]:
    rt = Runtime(settings, harness, **kw)
    try:
        yield await rt.open(migrate_db)
    finally:
        await rt.close()


QS_LINT = "lint_question"
LINT_SUBJECT = "registry_question"


def jev_lint_fn(jev: JevService, db: Database | None):
    """§7.4: `QS.lint_question` flags questions asking for counting, arithmetic or date comparison."""

    async def flags(qs: Any, question: str, text: str) -> bool:
        result = await jev.ask(QS_LINT, {"question": text}, LINT_SUBJECT, f"{qs.ref}.{question}")
        if db is not None:
            async with db.tx() as conn:
                await write_decisions(conn, result)
        return result.single.outcome == ACCEPT

    return flags
