"""S8 output rules (Q&A spec §1.4): a generated example that breaks a rule goes back to the harness with
the broken rule; a rewrite that passes is accepted, one the harness never fixes is rejected, and every
round's rule check is part of the example's provenance."""

from __future__ import annotations

import json

import pytest

from jev_graph_builder.pipeline.common import RunOptions, run_stage
from jev_graph_builder.pipeline.stages import all_stages
from tests.integration.fakes import NO

pytestmark = pytest.mark.integration


def _fails_source_reference(qs: str, question: str, state: object) -> float | None:
    """Jev for the rule: a question that points at "the passage" breaks `prompt_no_source_reference`."""
    if qs == "output_rules" and question == "prompt_no_source_reference" and "the passage" in json.dumps(state):
        return NO
    return None


async def _qa_single(ctx) -> list[dict]:
    rows = await ctx.db.fetch("SELECT payload, status, decision_ids FROM training_examples WHERE template = 'qa_single'")
    out = []
    for r in rows:
        checks = await ctx.db.fetch("SELECT decision_id, outcome FROM decisions WHERE decision_id = ANY(%s) "
                                    "AND question_set LIKE 'output_rules@%%'", (r["decision_ids"],))
        outcome = {c["decision_id"]: c["outcome"] for c in checks}
        # one entry per authoring round; an unchanged rewrite reuses the cached decision, so an id may repeat
        out.append({**r, "rule_checks": [outcome[d] for d in r["decision_ids"] if d in outcome]})
    return out


def _enable(ctx, *names: str) -> None:
    """Build these templates on top of the corpus defaults (as `train build NAME...` does)."""
    ctx.cache["train_only"] = [*(ctx.reg.corpus.get("training_templates") or []), *names]


async def _run_all(ctx) -> None:
    for stage in all_stages():
        rep = await run_stage(ctx, stage, RunOptions())
        assert not rep.errors, (stage.name, rep.errors[:3])


async def test_an_example_that_breaks_a_rule_is_rewritten_and_accepted(ctx, env) -> None:
    env.fake_jev.judge = _fails_source_reference
    await _run_all(ctx)
    examples = await _qa_single(ctx)
    accepted = [e for e in examples if e["status"] == "accepted"]
    assert accepted, examples[:3]
    for e in accepted:
        assert "the passage" not in e["payload"]["prompt"], e["payload"]
        # the first draft failed the rules, the rewrite passed; both checks are provenance
        assert e["rule_checks"] == ["reject", "accept"], e["rule_checks"]


async def test_an_example_the_harness_never_fixes_is_rejected(ctx, env) -> None:
    env.fake_jev.judge = _fails_source_reference
    env.harness.ignore_revisions = True
    await _run_all(ctx)
    examples = await _qa_single(ctx)
    assert examples
    assert not [e for e in examples if e["status"] == "accepted"], examples[:3]
    rounds = ctx.reg.policy("training.output_rules.reauthor_rounds")
    for e in examples:
        if e["rule_checks"]:  # an example rejected by an earlier filter never reached the rule check
            assert e["status"] == "rejected" and e["rule_checks"] == ["reject"] * (rounds + 1), (e["status"], e["rule_checks"])


async def test_code_exported_examples_carry_the_text_they_name(ctx) -> None:
    _enable(ctx, "triples", "retrieval_pairs")
    await _run_all(ctx)
    rows = await ctx.db.fetch("SELECT payload FROM training_examples WHERE template = 'triples' AND status = 'accepted'")
    assert rows
    for r in rows:
        p = r["payload"]
        ends = [p["subject"], p["object"]] if "predicate" in p else [p["source"]]
        assert all(e["text"] for e in ends), p
        if "predicate" in p:
            assert p["predicate_definition"], p
    pairs = await ctx.db.fetch("SELECT payload FROM training_examples WHERE template = 'retrieval_pairs' AND status = 'accepted'")
    for r in pairs:
        assert all(c["text"] for c in [*r["payload"]["positives"], *r["payload"]["negatives"]]), r["payload"]
        texts = [" ".join(c["text"].split()) for c in [*r["payload"]["positives"], *r["payload"]["negatives"]]]
        assert len(texts) == len(set(texts)), r["payload"]  # no negative repeats another passage of the pair
    facts = await ctx.db.fetch("SELECT payload FROM training_examples WHERE template = 'facts' AND status = 'accepted'")
    assert facts
    assert all(r["payload"]["claim"] and r["payload"]["source"]["text"] and "predicate" not in r["payload"] for r in facts)


async def test_qa_covers_every_verified_fact(ctx) -> None:
    await _run_all(ctx)
    cap = ctx.reg.policy("training.coverage.facts_per_chunk")
    claims = await ctx.db.fetch("SELECT cl.chunk_id, cl.text FROM claims cl JOIN chunks c ON c.chunk_id = cl.chunk_id "
                                "WHERE cl.status = 'accepted' AND c.status = 'accepted' AND NOT coalesce(c.boilerplate, false)")
    assert claims
    per_chunk: dict[str, set[str]] = {}
    for c in claims:
        per_chunk.setdefault(c["chunk_id"], set()).add(c["text"])
    from jev_graph_builder.pipeline.s8_training import TrainingStage

    asked: dict[str, set[str]] = {}
    for i in await TrainingStage().enumerate(ctx):
        if i.payload["template"] == "qa_single":
            src = i.payload["source"]
            asked.setdefault(src["chunk_id"], set()).add(src.get("fact"))
    for chunk_id, texts in per_chunk.items():
        assert len(asked.get(chunk_id, set())) == min(len(texts), cap), (chunk_id, texts, asked.get(chunk_id))
        assert asked[chunk_id] <= texts, (chunk_id, asked[chunk_id])


async def test_a_question_that_brings_in_unstated_facts_is_rejected(ctx, env) -> None:
    env.fake_jev.judge = lambda qs, question, state: NO if qs == "train_premise" else None
    await _run_all(ctx)
    rows = await ctx.db.fetch("SELECT template, status FROM training_examples WHERE template IN ('qa_single', 'qa_multihop')")
    assert rows
    assert not [r for r in rows if r["status"] == "accepted"], rows


async def test_a_sampled_run_derives_from_every_accepted_example(ctx) -> None:
    from jev_graph_builder.pipeline.s8_training import DerivedTrainingStage

    _enable(ctx, "retrieval_pairs")
    stages = all_stages()
    for stage in stages:
        if stage.name != "training_derived":
            rep = await run_stage(ctx, stage, RunOptions())
            assert not rep.errors, (stage.name, rep.errors[:3])
    sources = await ctx.db.fetch("SELECT count(*) AS n FROM training_examples WHERE template = ANY(%s) AND status = 'accepted'",
                                 (ctx.reg.policy("training.derive_from"),))
    assert sources[0]["n"]
    # `--sample` picks among generated sources; derived examples are built from whatever that produced
    rep = await run_stage(ctx, DerivedTrainingStage(), RunOptions(sample=1e-9, dry_run=True))
    templates = [t for t in ctx.cache["train_only"] if ctx.reg.training_template(t)["kind"] == "retrieval_pairs"]
    assert templates
    assert rep.selected == sources[0]["n"] * len(templates), (rep.selected, sources[0]["n"], templates)


async def test_an_example_whose_source_is_no_longer_selected_is_superseded(ctx) -> None:
    from jev_graph_builder.pipeline.s8_training import TrainingStage

    await _run_all(ctx)
    async with ctx.db.tx() as conn:
        await conn.execute("INSERT INTO training_examples (example_id, template, payload, status, registry_version) "
                           "SELECT 'gone', template, jsonb_set(payload, '{source_key}', '\"gone\"'), status, registry_version "
                           "FROM training_examples WHERE template = 'qa_single' AND status = 'accepted' LIMIT 1")
    rep = await run_stage(ctx, TrainingStage(), RunOptions())
    assert not rep.errors, rep.errors[:3]
    rows = await ctx.db.fetch("SELECT example_id, status FROM training_examples WHERE example_id = 'gone' "
                              "OR (template = 'qa_single' AND status = 'accepted')")
    assert {r["example_id"]: r["status"] for r in rows}.get("gone") == "superseded", rows
    assert len(rows) > 1, rows  # still-selected examples are kept


async def test_retrieval_negatives_come_from_the_positives_split(ctx) -> None:
    _enable(ctx, "retrieval_pairs")
    await _run_all(ctx)
    rows = await ctx.db.fetch("SELECT split, source_chunk_ids FROM training_examples WHERE template = 'retrieval_pairs' AND status = 'accepted'")
    assert rows
    assert all(r["split"] != "cross_split" for r in rows), [r["split"] for r in rows]
