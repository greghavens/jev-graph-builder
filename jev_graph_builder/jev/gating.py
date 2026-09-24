"""Gating engine (§11.5, P4).

A question set's `gating.accept_when` names which of Jev's answers mean "accept":

- Noul: Jev's answer counts when Jev's probability for it is above the threshold:
  `policies.gating.threshold`, or the question set's or the leaf's own `threshold`.
  Yes is p > t; no is 1 - p > t.
- Choice: the option Jev chose.
- Score: the level Jev rated most probable.

Back-off (as in jev-no-bullshit): when Jev's answers to a question send the same
subject round a loop and that question is asked again (e.g. S0 disambiguation
re-checks the redrafted relation vocabulary for overlaps), the threshold keeps only
`policies.gating.backoff` of its remaining distance to 1 per repeat:
t_k = 1 - (1 - t) * backoff^k, so 0.5, 0.75, 0.875 with the defaults.

A question set with no `accept_when` (a pure Choice such as a routing or naming
question) is always accepted: Jev's pick is the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jev_graph_builder.registry.loader import QuestionSet, RegistryError

ACCEPT, REJECT = "accept", "reject"
YES, NO = "yes", "no"


@dataclass
class GateResult:
    outcome: str


@dataclass(frozen=True)
class Bar:
    """How sure Jev must be of a Noul answer for it to count."""

    threshold: float  # policies.gating.threshold
    backoff: float    # policies.gating.backoff: share of the remaining distance to 1 kept per repeat
    repeat: int       # times this question already came back to Jev for the same subject

    def __post_init__(self) -> None:
        if not 0 <= self.threshold < 1:
            raise RegistryError(f"gating threshold {self.threshold} must be in [0, 1)")
        if not 0 < self.backoff <= 1:
            raise RegistryError(f"gating backoff {self.backoff} must be in (0, 1]")
        if self.repeat < 0:
            raise ValueError(f"gating repeat {self.repeat} must be >= 0")

    def at(self, base: float) -> float:
        """`base` (a policy, question-set or leaf threshold) after this bar's back-off."""
        return 1 - (1 - base) * self.backoff ** self.repeat

    def base_for(self, qs: QuestionSet) -> float:
        """The question set's own `gating.threshold`, else the policy threshold."""
        return (qs.gating or {}).get("threshold", self.threshold)

    def for_qs(self, qs: QuestionSet) -> float:
        return self.at(self.base_for(qs))

    def for_leaves(self, qs: QuestionSet) -> dict[str, float]:
        """Noul questions whose `accept_when` leaf sets its own threshold, after back-off."""
        return {c["q"]: self.at(c["threshold"]) for c in _leaves((qs.gating or {}).get("accept_when"))
                if "threshold" in c}


def _leaves(cond: dict | None) -> list[dict]:
    if cond is None:
        return []
    for key in ("all", "any"):
        if key in cond:
            return [leaf for c in cond[key] for leaf in _leaves(c)]
    if "not" in cond:
        return _leaves(cond["not"])
    return [cond]


def says_yes(answer: dict[str, Any], threshold: float) -> bool:
    """Jev's yes on a Noul: its probability that the statement is true is above `threshold`."""
    return float(answer["p"]) > threshold


def says_no(answer: dict[str, Any], threshold: float) -> bool:
    """Jev's no on a Noul: its probability that the statement is false is above `threshold`."""
    return 1 - float(answer["p"]) > threshold


def top_level(answer: dict[str, Any]) -> int:
    """The level Jev rated most probable on a Score (its mode; never an interpolated value)."""
    probs = answer.get("probabilities") or {}
    if not probs:
        return round(float(answer["score"]))
    return int(max(sorted(probs), key=lambda k: float(probs[k])))


class _Skipped:
    """A leaf marked `if_asked` whose question was not asked for this subject: ignored by `all`/`any`."""

    def __repr__(self) -> str:
        return "SKIPPED"


SKIPPED = _Skipped()


def _and(values: list[Any]) -> Any:
    values = [v for v in values if v is not SKIPPED]
    return all(values) if values else SKIPPED


def _or(values: list[Any]) -> Any:
    values = [v for v in values if v is not SKIPPED]
    return any(values) if values else SKIPPED


def _leaf(cond: dict, answers: dict[str, dict], bar: Bar, qs_threshold: float) -> Any:
    q = cond["q"]
    a = answers.get(q)
    if a is None and cond.get("if_asked"):
        return SKIPPED
    if a is None:
        raise RegistryError(f"gating refers to `{q}`, which has no answer")
    if a["type"] == "noul":
        t = bar.at(cond.get("threshold", qs_threshold))
        return says_yes(a, t) if cond["is"] == YES else says_no(a, t)
    if a["type"] == "choice":
        if "eq" in cond:
            return a["choice"] == cond["eq"]
        if "in" in cond:
            return a["choice"] in cond["in"]
        return a["choice"] not in cond["not_in"]
    return top_level(a) in cond["level_in"]


def _eval(cond: dict, answers: dict[str, dict], bar: Bar, qs_threshold: float) -> Any:
    if "all" in cond:
        return _and([_eval(c, answers, bar, qs_threshold) for c in cond["all"]])
    if "any" in cond:
        return _or([_eval(c, answers, bar, qs_threshold) for c in cond["any"]])
    if "not" in cond:
        r = _eval(cond["not"], answers, bar, qs_threshold)
        return r if r is SKIPPED else not r
    return _leaf(cond, answers, bar, qs_threshold)


def evaluate(qs: QuestionSet, answers: dict[str, dict], bar: Bar) -> GateResult:
    cond = (qs.gating or {}).get("accept_when")
    if cond is None:
        return GateResult(ACCEPT)
    # SKIPPED: every gating question was `if_asked` and none was asked for this subject — Jev rejected nothing.
    result = _eval(cond, answers, bar, bar.base_for(qs))
    return GateResult(ACCEPT if result is True or result is SKIPPED else REJECT)
