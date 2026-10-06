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


async def test_a_sampled_run_derives_from_every_accepted_example(ctx) -> None:
    from jev_graph_builder.pipeline.s8_training import DerivedTrainingStage

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
    templates = [t for t in ctx.reg.corpus.get("training_templates") or [] if ctx.reg.training_template(t)["kind"] == "retrieval_pairs"]
    assert templates
    assert rep.selected == sources[0]["n"] * len(templates), (rep.selected, sources[0]["n"], templates)
