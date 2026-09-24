"""Registry loading/lint/versioning, question rendering (opaque keys, fan-out) and state building (§7, §11)."""
from __future__ import annotations

import pytest
import yaml

from jev_graph_builder.jev.questions import QuestionBuilder
from jev_graph_builder.jev.state import StateBuilder, StateError
from jev_graph_builder.jev.tokens import TokenEstimator
from jev_graph_builder.registry.lint import lint_question_set, lint_registry
from jev_graph_builder.registry.loader import QuestionSet, Registry, RegistryError
from jev_graph_builder.registry.versioning import activate, diff_trees, get_proposal, stage_proposal, start_draft


def limits(reg):
    return reg.profile("jev", reg.policy("jev.limits_profile"))["limits"]


def test_seed_registry_lints_clean(reg):
    rep = lint_registry(reg)
    assert rep.errors == [] and rep.warnings == []


def test_policy_has_no_defaults(reg):
    with pytest.raises(RegistryError):
        reg.policy("link.does_not_exist")


def test_opaque_keys_and_no_logical_names_on_wire(reg):
    qs = reg.question_set("link")
    r = QuestionBuilder(reg, limits(reg)).render(qs)
    assert set(r.wire) == set(r.keys)
    for key, info in r.keys.items():
        assert key.startswith("k") and info.logical not in key
    # Keys are stable across renders (cacheable) and the questions hash covers the wire form only.
    assert QuestionBuilder(reg, limits(reg)).render(qs).questions_hash == r.questions_hash
    relation = next(b for k, b in r.wire.items() if r.keys[k].logical == "relation")
    assert {"none", "other", "depends_on"} <= set(relation["criteria"])


def test_fanout_substitutes_item_reference(reg):
    qs = reg.question_set("link_fanout")
    r = QuestionBuilder(reg, limits(reg)).render(qs, items=["c1", "c2"])
    per_item = [b["instructions"] for k, b in r.wire.items() if r.keys[k].item == "c2"]
    assert per_item and all("`candidates.c2`" in t or "c2" in t for t in per_item)
    assert all("{{" not in b["instructions"] for b in r.wire.values())


def test_state_wraps_untrusted_and_refuses_numbers(reg):
    sb = StateBuilder(reg, TokenEstimator(reg.profile("jev", "typesafe")["chars_per_token"]))
    qs = reg.question_set("summary_verify")
    field = reg.policy("jev.untrusted_field")
    state = sb.build(qs, {"chunk": "The gateway retries twice.", "sentence": "It retries.", "task": "Summarise the chunk.",
                          "context": {"document_title": "Gateway guide"}})
    untrusted = [k for k, s in qs.state_template.items() if s.get("untrusted")]
    assert untrusted and all(set(state[k]) == {field} for k in untrusted)
    with pytest.raises(StateError):
        sb.build(qs, {"chunk": "x", "sentence": 3, "task": "t"})
    with pytest.raises(StateError):
        sb.build(qs, {"sentence": "x", "task": "t"})


def _qs(raw) -> QuestionSet:
    return QuestionSet(name=raw["name"], version=raw["version"], jev_model=raw["jev_model"],
                       state_template=raw["state_template"], questions=raw["questions"], gating=raw["gating"],
                       fanout=raw.get("fanout"), content_hash="x", raw=raw)


def test_lint_catches_authoring_mistakes(reg):
    raw = {
        "name": "bad", "version": 1, "jev_model": "jev-1.13.0", "purpose": "p",
        "state_template": {"chunk": {}},
        "questions": {
            "a": {"type": "noul", "instructions": "Is `chunk` about `b`?", "criteria": {"true": "It is not about it", "false": "x"}},
            "b": {"type": "score", "instructions": "How dense is `chunk`?", "criteria": ["only"]},
            "c": {"type": "noul", "instructions": "For {{ item }}: is it relevant?"},
        },
        "gating": {"accept_when": {"q": "zzz", "gte": "$cal.zzz.accept"}},
    }
    errors = " | ".join(lint_question_set(reg, _qs(raw), limits(reg)).errors)
    assert "question key `b`" in errors
    assert "negation" in errors
    assert "levels" in errors
    assert "zzz" in errors
    assert "item" in errors


def test_proposal_stage_diff_activate(reg_copy):
    reg = reg_copy
    draft = start_draft(reg.root, reg.root.parent / "scratch")
    pol = yaml.safe_load((draft / "policies.yaml").read_text())
    pol["search"]["k"] = pol["search"]["k"] + 1
    (draft / "policies.yaml").write_text(yaml.safe_dump(pol, sort_keys=False))
    prop = stage_proposal(reg.root, draft)
    assert prop.version != reg.version
    assert "policies.yaml" in diff_trees(reg.root, prop.path)
    assert Registry(reg.root).version == reg.version            # staging never touches the active tree
    new = activate(reg.root, get_proposal(reg.root, prop.version[:12]))
    assert new.version == prop.version and new.policy("search.k") == pol["search"]["k"]


def test_chunk_metadata_carries_only_policy_fields_as_text(tmp_path) -> None:
    """Metadata reaches Jev state only for keys the approved policy names, and never as numbers (§11.4)."""
    import shutil
    from pathlib import Path

    from jev_graph_builder.pipeline.common import chunk_metadata

    root = tmp_path / "registry"
    shutil.copytree(Path(__file__).resolve().parents[2] / "registry", root)
    pol = yaml.safe_load((root / "policies.yaml").read_text())
    pol["ingest"]["metadata_fields"] = ["release", "tags"]
    (root / "policies.yaml").write_text(yaml.safe_dump(pol))
    reg = Registry(root)
    meta = {"release": 9.1, "tags": ["a", 2], "token_count": 812, "source": "x"}
    assert chunk_metadata(reg, meta) == {"release": "9.1", "tags": ["a", "2"]}
    assert chunk_metadata(reg, {"token_count": 812}) is None
    assert chunk_metadata(reg, None) is None


def test_lint_requires_prompt_verifiers_to_be_active(reg_copy):
    """P3: a verifier that exists on disk but is not active would fail only when the stage runs."""
    from jev_graph_builder.registry.lint import lint_prompts

    assert not lint_prompts(reg_copy).errors
    path = reg_copy.root / "policies.yaml"
    pol = yaml.safe_load(path.read_text())
    del pol["question_sets"]["active"]["metadata_field"]
    path.write_text(yaml.safe_dump(pol))
    errors = lint_prompts(Registry(reg_copy.root)).errors
    assert any("metadata_field" in e and "active" in e for e in errors), errors


def test_fanout_asks_each_item_only_its_own_questions(reg):
    """Entities and claims share one call; each item is asked only the questions listed for it."""
    qs = reg.question_set("extract_verify_fanout")
    only = {"e": {"name_wrong", "type_wrong"}, "c": {"text_wrong", "supported", "type_wrong"}}
    r = QuestionBuilder(reg, limits(reg)).render(qs, items=["e", "c"], item_only=only, dynamic={"types": {"a": "A.", "b": "B."}})
    asked = {(k.item, k.logical) for k in r.keys.values()}
    assert asked == {(item, q) for item, qs_ in only.items() for q in qs_}


def test_lint_rejects_conflicting_leaf_thresholds(reg):
    raw = {
        "name": "t", "version": 1, "jev_model": "jev-1.13.0", "purpose": "p",
        "state_template": {"chunk": {}},
        "questions": {"a": {"type": "noul", "instructions": "Is `chunk` relevant?"}},
        "gating": {"accept_when": {"any": [{"q": "a", "is": "yes", "threshold": 0.6},
                                           {"q": "a", "is": "yes", "threshold": 0.8}]}},
    }
    assert any("conflicting leaf thresholds" in e for e in lint_question_set(reg, _qs(raw), limits(reg)).errors)
    raw["gating"]["accept_when"]["any"][1]["threshold"] = 0.6
    assert not any("conflicting" in e for e in lint_question_set(reg, _qs(raw), limits(reg)).errors)


def test_every_prompt_reads_as_a_task_for_jev(reg):
    """Jev is shown a job's task: every prompt renders, each template variable standing as its name."""
    for path in sorted((reg.root / "prompts").glob("*.md")):
        task = reg.prompt(path.stem).task
        assert task and "{{" not in task and "{%" not in task, path.name
    assert "`input_file`" in reg.prompt("extract").task


def test_verification_state_holds_the_whole_chunk_its_context_and_task(reg):
    """Every enrichment verifier sees the whole chunk (never cut), what the extractor saw with it, and its task."""
    from types import SimpleNamespace

    from jev_graph_builder.pipeline.s3_enrich import PROMPT, EnrichStage

    sb = StateBuilder(reg, TokenEstimator(reg.profile("jev", "typesafe")["chars_per_token"]))
    field = reg.policy("jev.untrusted_field")
    chunk = "The gateway retries twice on a timeout. " * 2000
    row = {"chunk_id": "c1", "text": chunk, "doc_title": "Gateway guide", "heading_path": ["Gateway", "Retries"],
           "meta": {"version": 9.1}}
    base = EnrichStage()._state(SimpleNamespace(reg=reg), row)
    extras = {"extract_verify": {"item": {"name": "gateway"}}, "extract_verify_fanout": {},
              "summary_verify": {"sentence": "It retries."}, "summary_verify_fanout": {}, "span_locate": {"span": "retries"}}
    for name, extra in extras.items():
        qs = reg.question_set(name)
        items = {"i0": {"name": "gateway"}} if qs.fanout else None
        state = sb.build(qs, {**base, **extra}, fanout_items=items)
        assert state["chunk"][field] == chunk, name
        assert state["context"][field] == {"document_title": "Gateway guide",
                                           "heading_path": reg.policy("segment.heading_separator").join(["Gateway", "Retries"]),
                                           "metadata": {"version": "9.1"}}, name
        assert state["task"] == reg.prompt(PROMPT).task, name
    bare = EnrichStage()._state(SimpleNamespace(reg=reg), {"chunk_id": "c2", "text": "x", "heading_path": [], "meta": None})
    assert "context" not in sb.build(reg.question_set("summary_verify"), {**bare, "sentence": "x"})
