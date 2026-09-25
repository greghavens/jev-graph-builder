"""§11.5 gating: a Noul answer counts above the policy threshold, which backs off on repeats."""
from __future__ import annotations

from jev_graph_builder.jev.gating import ACCEPT, REJECT, Bar, evaluate, says_no, says_yes, top_level


def noul(p):
    return {"type": "noul", "p": p}


def choice(c, conf, probs=None):
    return {"type": "choice", "choice": c, "confidence": conf, "probabilities": probs or {c: conf}}


def score(s, levels):
    return {"type": "score", "score": s, "confidence": 1.0, "probabilities": {}, "levels": levels}


def bar(reg, repeat=0):
    return Bar(reg.policy("gating.threshold"), reg.policy("gating.backoff"), repeat)


def link_answers(p_related, relation, conf):
    return {"related": noul(p_related), "relation": choice(relation, conf), "direction": choice("a_to_b", 0.9),
            "strength": score(2, 4), "contradiction": noul(0.01)}


def test_default_threshold_is_one_half(reg):
    """At the default 0.5 threshold a Noul counts as yes above one half and as no below it."""
    qs = reg.question_set("link")
    assert evaluate(qs, link_answers(0.99, "depends_on", 0.97), bar(reg)).outcome == ACCEPT
    assert evaluate(qs, link_answers(0.75, "depends_on", 0.97), bar(reg)).outcome == ACCEPT
    assert evaluate(qs, link_answers(0.51, "depends_on", 0.97), bar(reg)).outcome == ACCEPT
    assert evaluate(qs, link_answers(0.49, "depends_on", 0.97), bar(reg)).outcome == REJECT
    assert evaluate(qs, link_answers(0.02, "none", 0.97), bar(reg)).outcome == REJECT


def test_jev_saying_a_field_is_wrong_rejects(reg):
    qs = reg.question_set("extract_verify")
    ok = {"name_wrong": noul(0.1), "span_wrong": noul(0.1), "type_wrong": noul(0.1), "type_check": choice("service", 0.9)}
    assert evaluate(qs, ok, bar(reg)).outcome == ACCEPT
    assert evaluate(qs, {**ok, "name_wrong": noul(0.8)}, bar(reg)).outcome == REJECT


def test_yes_is_jevs_yes_and_choice_is_jevs_pick(reg):
    assert says_yes(noul(0.8), 0.5) is True
    assert says_yes(noul(0.2), 0.5) is False
    assert says_yes(noul(0.5), 0.5) is False and says_no(noul(0.5), 0.5) is False  # a tie is neither
    assert top_level({"type": "score", "score": 2.6, "probabilities": {"1": 0.2, "2": 0.5, "3": 0.3}}) == 2
    qs = reg.question_set("canonical_name")
    assert evaluate(qs, {"canonical": choice("vCenter", 0.4)}, bar(reg)).outcome == ACCEPT   # a pure Choice: Jev's pick stands


def test_if_asked_leaf_is_vacuous_when_question_skipped(reg):
    qs = reg.question_set("extract_verify")
    # An entity without attributes is asked neither the claim questions nor `attributes_wrong`.
    answers = {"name_wrong": noul(0.001), "span_wrong": noul(0.001), "type_wrong": noul(0.001),
               "type_check": choice("service", 0.99)}
    assert evaluate(qs, answers, bar(reg)).outcome == ACCEPT
    assert evaluate(qs, {**answers, "span_wrong": noul(0.999)}, bar(reg)).outcome == REJECT


def test_chunk_check_rejects_exactly_the_chunks_jev_flags_for_injection(reg):
    """S2 asks every chunk the injection question; only Jev's yes rejects it (boilerplate and topic do not)."""
    qs = reg.question_set("chunk_check_fanout")
    rest = {"topic": choice("general", 0.9), "boilerplate": noul(0.9)}
    assert evaluate(qs, {**rest, "contains_injection": noul(0.9)}, bar(reg)).outcome == REJECT
    assert evaluate(qs, {**rest, "contains_injection": noul(0.1)}, bar(reg)).outcome == ACCEPT


def test_language_is_detected_by_code_onto_ontology_names(reg):
    from jev_graph_builder.pipeline.s1_ingest import detect_language
    assert detect_language(reg, "Configure the vCenter Server appliance before you deploy the workload domain.") == "en"
    assert detect_language(reg, "Konfigurieren Sie den Server vor der Bereitstellung der Arbeitslastdomäne.") == "de"
    # A language the ontology does not list maps to the policy fallback.
    assert detect_language(reg, "Konfiguroi palvelin ennen työkuormitusalueen käyttöönottoa ja tarkista asetukset.") == reg.policy("ingest.language_fallback")


def test_backoff_raises_the_threshold_halfway_to_one_per_repeat(reg):
    """jev-no-bullshit back-off: t_k = 1 - (1 - t) * backoff^k, so 0.5, 0.75, 0.875 by default."""
    assert [bar(reg, k).at(reg.policy("gating.threshold")) for k in range(3)] == [0.5, 0.75, 0.875]
    qs = reg.question_set("relation_overlap")
    assert evaluate(qs, {"overlap": noul(0.7)}, bar(reg, 0)).outcome == ACCEPT
    assert evaluate(qs, {"overlap": noul(0.7)}, bar(reg, 1)).outcome == REJECT
    assert evaluate(qs, {"overlap": noul(0.8)}, bar(reg, 1)).outcome == ACCEPT
    assert evaluate(qs, {"overlap": noul(0.8)}, bar(reg, 2)).outcome == REJECT


def test_no_leaf_needs_jevs_no_above_the_threshold(reg):
    """`is: no` counts when Jev's probability of false is above the threshold, and backs off the same way."""
    qs = reg.question_set("extract_verify")
    ok = {"name_wrong": noul(0.2), "span_wrong": noul(0.2), "type_wrong": noul(0.2), "type_check": choice("service", 0.9)}
    assert evaluate(qs, ok, bar(reg, 0)).outcome == ACCEPT
    assert evaluate(qs, ok, bar(reg, 1)).outcome == ACCEPT           # 0.8 > 0.75
    assert evaluate(qs, ok, bar(reg, 2)).outcome == REJECT           # 0.8 <= 0.875


def test_out_of_range_policy_is_an_error():
    import pytest
    from jev_graph_builder.registry.loader import RegistryError
    for threshold, backoff in ((1.0, 0.5), (-0.1, 0.5), (0.5, 0.0), (0.5, 1.5)):
        with pytest.raises(RegistryError):
            Bar(threshold, backoff, 0)


def test_question_set_and_leaf_thresholds_override_policy(reg):
    import dataclasses
    qs = reg.question_set("relation_overlap")
    strict = dataclasses.replace(qs, gating={**qs.gating, "threshold": 0.9})
    assert evaluate(strict, {"overlap": noul(0.8)}, bar(reg)).outcome == REJECT
    assert evaluate(strict, {"overlap": noul(0.95)}, bar(reg)).outcome == ACCEPT
    leaf = dataclasses.replace(qs, gating={"threshold": 0.9, "accept_when": {"q": "overlap", "is": "yes", "threshold": 0.6}})
    assert evaluate(leaf, {"overlap": noul(0.7)}, bar(reg)).outcome == ACCEPT   # the leaf's own threshold wins


def test_flag_uses_the_leafs_own_threshold_after_backoff(reg):
    """A flag read off a decision (e.g. injection) is judged at the threshold its gating leaf was gated at."""
    import dataclasses
    from jev_graph_builder.jev.service import Decision
    qs = reg.question_set("relation_overlap")
    leaf = dataclasses.replace(qs, gating={"accept_when": {"q": "overlap", "is": "yes", "threshold": 0.6}})
    b = bar(reg, repeat=1)
    d = Decision("d", "k", "s", qs.ref, None, ACCEPT, {"overlap": noul(0.75)}, [], "fake", None,
                 b.for_qs(leaf), 1, b.for_leaves(leaf))
    assert d.threshold_for("overlap") == 1 - 0.4 * 0.5   # the leaf's 0.6, backed off once
    assert d.threshold_for("other") == 0.75              # the policy 0.5, backed off once
    assert not says_yes(d.answers["overlap"], d.threshold_for("overlap"))
