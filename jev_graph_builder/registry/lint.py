"""Registry lint (§7.4, §12.3). Each rule is a structural check; the one
semantic check (does a question ask for counting / arithmetic / date math?) is
itself a Jev decision (`QS.lint_question`) and runs only when a Jev service is
supplied."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from jev_graph_builder.registry.loader import QuestionSet, Registry, RegistryError

_BACKTICK = re.compile(r"`([^`]+)`")
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_WORD = re.compile(r"[\w']+")


@dataclass
class LintReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: LintReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def _texts(value: Any) -> list[str]:
    """All strings inside an instruction/criteria value (str, dict, list)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _texts(v)]
    if isinstance(value, list):
        return [t for v in value for t in _texts(v)]
    return []


def _choice_options(reg: Registry, criteria: Any) -> dict[str, Any]:
    if isinstance(criteria, dict) and "$from_ontology" in criteria:
        opts = {i["name"]: i["definition"] for i in reg.ontology(criteria["$from_ontology"])}
        opts.update(criteria.get("add") or {})
        return opts
    if isinstance(criteria, dict) and "$from_profiles" in criteria:
        return {k: v.get("description") for k, v in reg.profiles[criteria["$from_profiles"]].items()}
    return dict(criteria or {})


def _gating_questions(cond: Any) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    if not isinstance(cond, dict):
        return out
    if "q" in cond:
        out.append((cond["q"], cond))
    for key in ("all", "any"):
        for sub in cond.get(key, []):
            out.extend(_gating_questions(sub))
    if "not" in cond:
        out.extend(_gating_questions(cond["not"]))
    return out


def lint_question_set(reg: Registry, qs: QuestionSet, limits: dict[str, Any]) -> LintReport:
    rep = LintReport()
    where = qs.ref
    markers = {m.lower() for m in reg.policy("lint.negation_markers")}
    fields = set(qs.state_template)
    if qs.fanout:
        fields.add(qs.fanout["over"])
    qnames = set(qs.questions)

    for name, q in qs.questions.items():
        loc = f"{where}.{name}"
        texts = _texts(q.get("instructions"))
        crit = q.get("criteria")
        if q["type"] == "choice":
            opts = _choice_options(reg, crit)
            if not opts:
                rep.errors.append(f"{loc}: choice needs criteria")
            if len(opts) > limits["choice_max_options"]:
                src = crit.get("$from_ontology") if isinstance(crit, dict) else None
                has_parents = bool(src) and all(i.get("parent") for i in reg.ontology(src))
                (rep.warnings if has_parents else rep.errors).append(
                    f"{loc}: {len(opts)} options > {limits['choice_max_options']}"
                    + ("; hierarchical choice will be used" if has_parents else "; ontology needs `parent` for a hierarchical choice")
                )
            descs = [json.dumps(v, sort_keys=True) for v in opts.values() if v is not None]
            if len(descs) != len(set(descs)):
                rep.errors.append(f"{loc}: two options share an identical description (contradictory criteria)")
            texts += _texts(opts)
        elif q["type"] == "score":
            levels = crit if isinstance(crit, list) else []
            if not limits["score_min_levels"] <= len(levels) <= limits["score_max_levels"]:
                rep.errors.append(f"{loc}: score needs {limits['score_min_levels']}..{limits['score_max_levels']} levels, has {len(levels)}")
            if len({json.dumps(level, sort_keys=True) for level in levels}) != len(levels):
                rep.errors.append(f"{loc}: duplicate score levels")
            texts += _texts(levels)
        elif q["type"] == "noul" and isinstance(crit, dict):
            true_words = {w.lower() for t in _texts(crit.get("true")) for w in _WORD.findall(t)}
            if true_words & markers:
                rep.errors.append(f"{loc}: criteria.true contains a negation marker {sorted(true_words & markers)}")
            if crit.get("true") is not None and crit.get("true") == crit.get("false"):
                rep.errors.append(f"{loc}: criteria.true equals criteria.false")
            texts += _texts(crit)

        for text in texts:
            for ref in _BACKTICK.findall(text):
                head = ref.split(".")[0].strip()
                if head in qnames:
                    rep.errors.append(f"{loc}: references question key `{head}`; keys are not sent to the model")
                elif head not in fields:
                    rep.errors.append(f"{loc}: references `{head}`, which is not in state_template")
            for ph in _PLACEHOLDER.findall(text):
                if not qs.fanout or ph != "item":
                    rep.errors.append(f"{loc}: unknown placeholder {{{{ {ph} }}}}")
        if q.get("fanout") and not qs.fanout:
            rep.errors.append(f"{loc}: fanout question in a question set without `fanout`")

    leaf_thresholds: dict[str, set[float]] = {}
    for qn, cond in _gating_questions((qs.gating or {}).get("accept_when")):
        if "threshold" in cond:
            leaf_thresholds.setdefault(qn, set()).add(cond["threshold"])
        if qn not in qnames:
            rep.errors.append(f"{where}.gating.accept_when: unknown question `{qn}`")
            continue
        qtype = qs.questions[qn]["type"]
        form = "is" if qtype == "noul" else "level_in" if qtype == "score" else None
        if form and form not in cond:
            rep.errors.append(f"{where}.gating.accept_when: `{qn}` is a {qtype}; use `{form}`")
        if qtype == "choice" and not any(k in cond for k in ("eq", "in", "not_in")):
            rep.errors.append(f"{where}.gating.accept_when: `{qn}` is a choice; use `eq`, `in` or `not_in`")
        if "threshold" in cond and qtype != "noul":
            rep.errors.append(f"{where}.gating.accept_when: `{qn}` is a {qtype}; only a noul takes a threshold")
    for qn, ts in leaf_thresholds.items():
        if len(ts) > 1:  # a flag read off the decision must know which one Jev's answer was gated at
            rep.errors.append(f"{where}.gating.accept_when: `{qn}` has conflicting leaf thresholds {sorted(ts)}")

    for fname, spec in qs.state_template.items():
        ref = spec.get("max_tokens_ref")
        if ref:
            try:
                reg.policy(ref)
            except RegistryError as exc:
                rep.errors.append(f"{where}.state_template.{fname}: {exc}")
    if qs.fanout:
        try:
            reg.policy(qs.fanout["max_ref"])
        except RegistryError as exc:
            rep.errors.append(f"{where}.fanout: {exc}")
    return rep


def lint_prompts(reg: Registry) -> LintReport:
    """§12.3: prompts whose output enters the graph or a training set must name a verifier (P3)."""
    rep = LintReport()
    qs_names = set(reg.question_set_names())
    active = set(reg.policy("question_sets.active"))
    for p in reg.all_prompts():
        enters = p.meta.get("enters", "none")
        verified_by = p.meta.get("verified_by")
        refs = verified_by if isinstance(verified_by, list) else ([verified_by] if verified_by else [])
        if enters in {"graph", "training"} and not refs:
            rep.errors.append(f"prompt {p.ref}: output enters the {enters} but `verified_by` is empty")
        for r in refs:
            if r.split("@")[0] not in qs_names:
                rep.errors.append(f"prompt {p.ref}: verified_by `{r}` is not a question set")
            elif r.split("@")[0] not in active:
                rep.errors.append(f"prompt {p.ref}: verified_by `{r}` is not in policies.question_sets.active")
        for key in ("output_schema", "record_schema"):
            ref = p.meta.get(key)
            if ref:
                try:
                    reg.schema(ref)
                except RegistryError as exc:
                    rep.errors.append(f"prompt {p.ref}: {exc}")
    return rep


def lint_registry(reg: Registry) -> LintReport:
    rep = LintReport()
    try:
        _ = reg.corpus, reg.policies, reg.profiles
        limits = reg.profile("jev", reg.policy("jev.limits_profile"))["limits"]
    except (RegistryError, KeyError) as exc:
        rep.errors.append(str(exc))
        return rep
    for qs in reg.all_question_sets():
        rep.extend(lint_question_set(reg, qs, limits))
    for name, ver in reg.policy("question_sets.active").items():
        try:
            reg.question_set(name, int(ver))
        except RegistryError as exc:
            rep.errors.append(str(exc))
    rep.extend(lint_prompts(reg))
    for tname in reg.training_template_names():
        try:
            reg.training_template(tname)
        except RegistryError as exc:
            rep.errors.append(str(exc))
    return rep


LintJev = Callable[[QuestionSet, str, str], Awaitable[bool]]


async def lint_with_jev(reg: Registry, flags_forbidden_task: LintJev) -> LintReport:
    """Semantic lint through Jev: reject questions that ask for counting, arithmetic or date comparison.

    `flags_forbidden_task(qs, question_name, rendered_text)` returns True when Jev flags it.
    """
    rep = LintReport()
    for qs in reg.all_question_sets():
        for name, q in qs.questions.items():
            text = "\n".join(_texts(q.get("instructions")))
            if await flags_forbidden_task(qs, name, text):
                rep.errors.append(f"{qs.ref}.{name}: Jev flags this question as counting/arithmetic/date comparison")
    return rep
