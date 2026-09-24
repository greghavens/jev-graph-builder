"""End-to-end pipeline on the 20-document fixture with fake Jev and harness (§19, §20).

One pipeline run backs several acceptance items, because the run is the slow part:
3 (edges passed link + link_verify), 4 (P3 audit = 0), 6 (search / ask with
checked citations), 7 (every enabled template, document-disjoint splits,
provenance), and the idempotent re-run half of 2/9.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from jev_graph_builder.audit.report import edge_verification_violations, p3_violations
from jev_graph_builder.pipeline.common import RunOptions, run_stage
from jev_graph_builder.pipeline.s8_training import export
from jev_graph_builder.pipeline.stages import all_stages
from jev_graph_builder.planner import build_plan
from jev_graph_builder.query.graphrag import ask
from jev_graph_builder.query.search import search

pytestmark = pytest.mark.integration


async def _count(ctx, table: str, where: str = "") -> int:
    row = await ctx.db.fetchone(f"SELECT count(*) AS n FROM {table} {where}")  # noqa: S608
    return row["n"]


async def _run_all(ctx) -> dict:
    reports = {}
    for stage in all_stages():
        rep = await run_stage(ctx, stage, RunOptions())
        reports[stage.name] = rep
        assert not rep.errors, (stage.name, rep.errors[:3])
    return reports


async def test_full_pipeline(ctx, env, tmp_path: Path) -> None:
    await _run_all(ctx)
    assert await _count(ctx, "documents") == 20
    assert await _count(ctx, "chunks", "WHERE status = 'accepted'") >= 20
    assert await _count(ctx, "entities", "WHERE status = 'accepted'") > 0
    assert await _count(ctx, "edges", "WHERE status = 'accepted' AND NOT structural") > 0

    # Item 3: every accepted semantic edge passed `link` (or `link_fanout`) and `link_verify`.
    assert await edge_verification_violations(ctx.reg, ctx.db, ctx.corpus_id) == []
    # Item 4: every harness-produced graph / training record has an accepting Jev verification.
    p3 = await p3_violations(ctx.reg, ctx.db, ctx.corpus_id, limit=3)
    assert p3["total"] == 0, p3
    assert p3["rules"], "the P3 audit must have rules to check"

    # Items 2/9 (idempotency half): a second run redoes nothing and pays for nothing.
    jev_before, harness_before = len(env.fake_jev.calls), len(env.harness.runs)
    calls_before = await _count(ctx, "jev_calls", "WHERE NOT drift")
    runs_before = await _count(ctx, "harness_runs")
    drift_before = await _count(ctx, "jev_calls", "WHERE drift")
    await _run_all(ctx)
    drift_new = await _count(ctx, "jev_calls", "WHERE drift") - drift_before
    assert await _count(ctx, "jev_calls", "WHERE NOT drift") == calls_before
    assert await _count(ctx, "harness_runs") == runs_before
    assert len(env.harness.runs) == harness_before
    # The only new Jev traffic is the per-run drift sample (§8.8), which bypasses the cache by design.
    assert len(env.fake_jev.calls) - jev_before == drift_new
    plan = await build_plan(ctx, all_stages())
    assert plan.pending == 0, [s.as_dict() for s in plan.stages if s.runnable]

    # Item 6: search and ask; every kept answer sentence passed the citation check.
    found = await search(ctx, "Which service stores its state in Kafka?")
    assert found["results"], found
    answer = await ask(ctx, "What does Courier depend on?")
    assert answer["answer"], answer
    ids = [s["decision_id"] for s in answer["answer"]]
    rows = await ctx.db.fetch("SELECT decision_id, question_set, outcome FROM decisions WHERE decision_id = ANY(%s)", (ids,))
    assert len(rows) == len(ids)
    assert all(r["outcome"] == "accept" and r["question_set"].startswith("citation_check@") for r in rows)
    assert all(s["citation_ids"] and set(s["citation_ids"]) <= set(answer["citations"]) for s in answer["answer"])

    # Item 7: every enabled template, document-disjoint splits, full provenance.
    out = tmp_path / "export"
    written = await export(ctx, out, "jsonl")
    templates = {Path(p).stem for p in (ctx.reg.root / "training" / "templates").glob("*.yaml")}
    assert {k.split("/")[0] for k in written} == templates
    chunk_doc = {r["chunk_id"]: r["doc_id"] for r in await ctx.db.fetch("SELECT chunk_id, doc_id FROM chunks")}
    doc_splits: dict[str, set[str]] = defaultdict(set)
    for info in written.values():
        for line in Path(info["path"]).read_text().splitlines():
            rec = json.loads(line)
            assert rec["split"] in ("train", "validation", "test"), rec["split"]
            assert rec["source_chunk_ids"] and rec["decision_ids"] and rec["registry_version"] and rec["jev_model"], rec
            for cid in rec["source_chunk_ids"]:
                doc_splits[chunk_doc[cid]].add(rec["split"])
    leaks = {d: s for d, s in doc_splits.items() if len(s) > 1}
    assert not leaks, leaks
