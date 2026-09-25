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


async def test_a_new_question_set_version_reuses_answers_to_unchanged_questions(settings, monkeypatch) -> None:
    """§11.7: a version that only removes a question is answered from the stored call on the same state,
    with no Jev call. A reworded question, a changed option list or a different document is asked again."""
    import yaml

    from jev_graph_builder.registry.loader import Registry
    from tests.integration.conftest import Harnessed
    from tests.integration.fakes import FakeHarness, FakeJev

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    qs_dir = settings.resolved_registry() / "question_sets"
    base = yaml.safe_load((qs_dir / "doc_triage@2.yaml").read_text(encoding="utf-8"))
    removed = {**base, "version": 3, "questions": {"in_scope": base["questions"]["in_scope"]}}
    reworded = {**removed, "version": 4, "questions": {"in_scope": {
        "type": "noul", "instructions": "`document` belongs in the corpus described in `corpus.scope`."}}}
    fewer_options = {**base, "version": 5, "questions": {**base["questions"], "doc_type": {
        **base["questions"]["doc_type"], "criteria": {"how_to": "Steps to do a task.", "reference": "Facts to look up."}}}}
    for doc in (removed, reworded, fewer_options):
        (qs_dir / f"doc_triage@{doc['version']}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    reg = Registry(settings.resolved_registry())
    env = Harnessed(settings, FakeJev(reg), FakeHarness(reg))
    rt, ctx = await env.open()
    try:
        inputs = {"document": {"title": "T", "headings": ["H"], "opening": "Some text."}, "corpus": {"scope": "docs"}}
        first = (await ctx.jev.ask("doc_triage", inputs, "document", "d1", qs_version=2)).single
        paid = len(env.fake_jev.calls)
        again = await ctx.jev.ask("doc_triage", inputs, "document", "d1", qs_version=3)
        assert len(env.fake_jev.calls) == paid and again.cache_hits == 1
        assert again.single.answers["in_scope"] == first.answers["in_scope"]
        assert again.single.outcome == first.outcome
        await ctx.jev.ask("doc_triage", inputs, "document", "d1", qs_version=4)
        assert len(env.fake_jev.calls) == paid + 1
        await ctx.jev.ask("doc_triage", inputs, "document", "d1", qs_version=5)
        assert len(env.fake_jev.calls) == paid + 2
        other = {**inputs, "document": {**inputs["document"], "opening": "Other text."}}
        await ctx.jev.ask("doc_triage", other, "document", "d2", qs_version=3)
        assert len(env.fake_jev.calls) == paid + 3
    finally:
        await rt.close()
