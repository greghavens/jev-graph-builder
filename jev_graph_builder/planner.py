"""`jev-graph-builder plan` (§14.3, §14.5, §18).

For every stage: how many items are missing / stale / failed / pending / done
against the active Registry and profiles, and the estimated Jev calls, Jev tokens
and wall-clock time of the runnable ones. Per-item figures are
Registry policy (`policies.plan.estimates.<stage>`), refined from the observed
averages in `jev_calls` / `harness_runs` once a stage has run. Counts for a
downstream stage reflect the upstream rows that exist now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jev_graph_builder.pipeline.common import RUNNABLE, Context, RunOptions, Stage, select_items

SECONDS_PER_MINUTE = 60


@dataclass
class StagePlan:
    stage: str
    by_status: dict[str, int]
    runnable: int
    jev_calls: float
    jev_tokens: float
    wall_clock_s: float
    basis: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Plan:
    stages: list[StagePlan] = field(default_factory=list)

    @property
    def pending(self) -> int:
        return sum(s.runnable for s in self.stages)

    def totals(self) -> dict[str, float]:
        return {
            "runnable_items": self.pending,
            "jev_calls": sum(s.jev_calls for s in self.stages),
            # Stages run in order, so the critical path is the sum of stage times.
            "wall_clock_s": sum(s.wall_clock_s for s in self.stages),
            "bottleneck": max(self.stages, key=lambda s: s.wall_clock_s).stage if self.stages else None,
        }


async def _observed(ctx: Context, stage: str) -> dict[str, float] | None:
    """Per-item averages from the last finished run of this stage, if any."""
    row = await ctx.db.fetchone(
        """
        WITH r AS (SELECT run_id FROM runs WHERE stage = %s AND status = 'done' ORDER BY ended_at DESC LIMIT 1)
        SELECT (SELECT count(*) FROM work_items w WHERE w.run_id = (SELECT run_id FROM r)) AS items,
               (SELECT count(*) FROM jev_calls j WHERE j.run_id = (SELECT run_id FROM r)) AS calls,
               (SELECT coalesce(sum((usage->>'input_tokens')::bigint), 0)::float FROM jev_calls j WHERE j.run_id = (SELECT run_id FROM r)) AS tokens
        """, (stage,))
    if not row or not row["items"]:
        return None
    n = row["items"]
    return {"jev_calls_per_item": row["calls"] / n, "tokens_per_call": row["tokens"] / max(row["calls"], 1)}


async def plan_stage(ctx: Context, stage: Stage) -> StagePlan:
    _, by_status, _ = await select_items(ctx, stage, RunOptions(dry_run=True))
    runnable = sum(n for s, n in by_status.items() if s in RUNNABLE)
    est = dict(ctx.reg.policy(f"plan.estimates.{stage.name}"))
    observed = await _observed(ctx, stage.name)
    basis = "policy"
    if observed:
        est.update(observed)
        basis = "observed"
    prof = ctx.jev.profile
    calls = runnable * est["jev_calls_per_item"]
    tokens = calls * est["tokens_per_call"]
    jev_s = calls / prof["rate"]["requests_per_minute"] * SECONDS_PER_MINUTE
    token_s = tokens / prof["rate"]["tokens_per_second"]
    harness_s = runnable * est["harness_seconds_per_item"] / ctx.reg.policy(f"run.concurrency.{stage.name}")
    return StagePlan(
        stage=stage.name, by_status=by_status, runnable=runnable, jev_calls=calls, jev_tokens=tokens,
        wall_clock_s=max(jev_s, token_s) + harness_s, basis=basis,
    )


async def build_plan(ctx: Context, stages: list[Stage]) -> Plan:
    plan = Plan()
    for stage in stages:
        plan.stages.append(await plan_stage(ctx, stage))
    return plan
