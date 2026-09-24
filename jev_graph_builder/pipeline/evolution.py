"""Ontology evolution (§8.7.4).

Tests Jev's `other` picks over the S6 link picks (`none` excluded) not yet
examined. During S6, evolution runs as soon as the `other` rate there is a
statistically significant excess over `policies.ontology.other_baseline_share`
(one-sided binomial test at `significance`). Once S6 has judged every pair, a
final pass takes whatever `other` picks remain unexamined. Either way a harness
job clusters the `other` pairs and proposes new relation types, and a type is
kept only when Jev's yes rate over its cited pairs is significantly above
`support_baseline_share` (same test): a recurring kind of link, not a few
one-offs. The proposals also go through the §8.1 overlap check. Jev decides:
the relations it accepts, and does not find overlapping an existing or
already-kept relation, are activated as a new Registry version at once. S6
depends on `relation_types`, so the build then re-asks its pairs under the new
vocabulary (§14.3). Examined picks are marked, so each attempt tests fresh data.

Jev's approval is written in the same transaction that marks the picks examined;
activation follows, and a build killed in between activates the approved row on
resume. After an activation, evolution stops for the rest of the phase: this
phase still holds the old vocabulary, and the build re-runs it under the new one.
"""

from __future__ import annotations

from typing import Any

import yaml
from scipy.stats import binomtest

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.ledger.ledger import DONE
from jev_graph_builder.pipeline.common import Context, Stage, run_single_item, write_decisions
from jev_graph_builder.pipeline.ontology_checks import CheckReport, check_definitions
from jev_graph_builder.registry import gate
from jev_graph_builder.registry.versioning import get_proposal, stage_proposal, start_draft
from jev_graph_builder.store import repo

PROMPT, SCHEMA = "propose_relations", "relation_proposals"
JEV_APPROVER = "jev"
RELATIONS = "relation_types"


def jev_accepted(proposals: list[dict[str, Any]], check: CheckReport) -> list[dict[str, Any]]:
    """Proposals Jev found supported, kept in name order unless Jev found one overlapping
    an existing relation or a proposal kept before it."""
    unsupported = {u["name"] for u in check.unsupported}
    overlapping = {frozenset((o["a"], o["b"])) for o in check.overlaps}
    new_names = {p["name"] for p in proposals}
    kept: list[dict[str, Any]] = []
    for p in sorted(proposals, key=lambda x: x["name"]):
        if p["name"] in unsupported:
            continue
        clashes = {n for pair in overlapping if p["name"] in pair for n in pair} - {p["name"]}
        if clashes - new_names or clashes & {k["name"] for k in kept}:
            continue
        kept.append(p)
    return kept


def significantly_above(k: int, n: int, share: float, alpha: float) -> bool:
    """One-sided binomial test: are k successes in n trials significantly more than `share`?"""
    return n > 0 and binomtest(k, n, share, alternative="greater").pvalue < alpha


def evolution_due(rels: list[str], pol: dict[str, Any], final: bool) -> tuple[bool, int, int]:
    """(due, other picks, picks) over the unexamined S6 picks, `none` excluded. Never before there are
    enough `other` pairs for a type to pass the support test even if Jev confirmed every one; then,
    during S6, only on a significant `other` excess, and in the final pass on what is left."""
    picks = [r for r in rels if r != pol["none_label"]]
    k, n = picks.count(pol["other_label"]), len(picks)
    alpha = pol["significance"]
    enough = significantly_above(k, k, pol["support_baseline_share"], alpha)
    return enough and (final or significantly_above(k, n, pol["other_baseline_share"], alpha)), k, n


LINK_WINDOW = ("corpus_id = %s AND NOT structural AND NOT evolution_seen AND src_kind = 'chunk' "
               "AND dst_kind = 'chunk' AND status <> 'superseded'")
EVOLVED = "ontology_evolved"
PENDING = ("SELECT version FROM registry_versions WHERE corpus_id = %s AND status = %s AND approved_by = %s "
           "AND summary->>'kind' = 'ontology_evolution' ORDER BY approved_at, version")


async def activate_pending(ctx: Context) -> str | None:
    """Activate Jev-approved evolution proposals not yet active (also finishes one cut short by a kill)."""
    version = None
    for row in await ctx.db.fetch(PENDING, (ctx.corpus_id, gate.APPROVED, JEV_APPROVER)):
        version = (await gate.activate(ctx, get_proposal(ctx.reg.root, row["version"])))["active"]
    return version


async def maybe_evolve_ontology(ctx: Context, stage: Stage, final: bool = False) -> str | None:
    """During S6 (`final` false): only on a significant `other` excess. Final pass: on any `other` left.
    Returns the newly active Registry version, if any."""
    if ctx.cache.get(EVOLVED):
        return None
    version = await activate_pending(ctx)
    if version is None:
        window = await ctx.db.fetch(f"SELECT edge_id, rel FROM edges WHERE {LINK_WINDOW} ORDER BY edge_id", (ctx.corpus_id,))
        due, k, n = evolution_due([w["rel"] for w in window], ctx.reg.policy("ontology"), final)
        if not due:
            return None
        window_ids = [w["edge_id"] for w in window]
        item = stage.item(ctx, f"evolve:{ctx.corpus_id}:{sha256_hex(*window_ids)}", window_ids)
        await run_single_item(ctx, stage, item, lambda: _propose(ctx, window, final, k, n))
        version = await activate_pending(ctx)
    if version is not None:
        ctx.cache[EVOLVED] = True
    return version


async def _propose(ctx: Context, window: list[dict[str, Any]], final: bool, k: int, n: int):
    """Harness proposes from the `other` pairs; Jev checks; returns the item's writer."""
    reg = ctx.reg
    pol = reg.policy("ontology")
    other_ids = [w["edge_id"] for w in window if w["rel"] == pol["other_label"]]
    examples = await ctx.db.fetch(
        "SELECT e.edge_id, a.text AS a, b.text AS b FROM edges e JOIN chunks a ON a.chunk_id = e.src_id "
        "JOIN chunks b ON b.chunk_id = e.dst_id WHERE e.edge_id = ANY(%s) ORDER BY e.edge_id LIMIT %s",
        (other_ids, pol["evolution_examples"]))
    existing = reg.ontology(RELATIONS)
    passages = {e["edge_id"]: (e["a"], e["b"]) for e in examples}  # (source, target): judged as one link
    ws = ctx.jobs.workspace_for(PROMPT, sha256_hex(sorted(passages)))
    ws.mkdir(parents=True, exist_ok=True)
    profile = await ctx.jobs.choose_profile(PROMPT, {"records": [f"{a}\n{b}" for a, b in passages.values()]})
    result, run_id = await ctx.jobs.run_single(
        PROMPT, SCHEMA, profile,
        {"examples": [{"id": i, "source": a, "target": b} for i, (a, b) in passages.items()], "existing": existing,
         "other_count": k, "picks": n}, ws)
    proposals = (result.structured or {}).get("relations", []) if result.ok else []

    def supported(yes: int, total: int) -> bool:
        return significantly_above(yes, total, pol["support_baseline_share"], pol["significance"])

    check = await check_definitions(ctx.jev, RELATIONS, [*existing, *proposals], passages,
                                    reg.policy("run.concurrency.ontology_checks"),
                                    only={p["name"] for p in proposals}, supported=supported)
    kept = jev_accepted(proposals, check)
    proposal, lint = None, []
    if kept:
        draft = start_draft(reg.root, ctx.jobs.root / "evolve" / "draft")
        (draft / "ontology" / f"{RELATIONS}.yaml").write_text(
            yaml.safe_dump({"items": [*existing, *kept]}, sort_keys=False, allow_unicode=True), encoding="utf-8")
        proposal = stage_proposal(reg.root, draft)
        lint = gate.lint_errors(proposal)
    summary = {"kind": "ontology_evolution", "final": final, "other_count": k, "picks": n, "proposals": proposals,
               "kept": [p["name"] for p in kept], "lint_errors": [str(e) for e in lint], "checks": check.as_dict(),
               "harness_run_id": run_id}

    async def write(conn) -> str:
        await write_decisions(conn, *check.decisions)
        await conn.execute("UPDATE edges SET evolution_seen = true WHERE edge_id = ANY(%s)", ([w["edge_id"] for w in window],))
        subject = proposal.version if proposal else None
        await repo.upsert(conn, "alerts", {"alert_id": sha256_hex("ontology_evolution", subject or "", ctx.run_id or ""),
                                           "kind": "ontology_evolution", "subject": subject, "payload": summary,
                                           "run_id": ctx.run_id}, key=("alert_id",))
        if proposal is not None and not lint:
            await gate.record_approval(conn, ctx.corpus_id, proposal, summary, JEV_APPROVER)
        return DONE

    return write
