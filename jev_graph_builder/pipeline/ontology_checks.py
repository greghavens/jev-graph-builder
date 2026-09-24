"""Jev checks on ontology drafts (§8.1 step 4, reused by §8.7.4).

* Every definition: Noul "the definition is supported by the cited sample
  passages" (`QS.ontology_support`). A relation type cites (source, target)
  passage pairs instead, and Jev judges each pair as one link
  (`QS.relation_support`): supported if any pair shows it, or by the caller's
  `supported(yes, total)` rule (§8.7.4 uses a significance test).
* Every pair of relation types: Noul "these definitions overlap so much that
  one chunk pair could satisfy both" (`QS.relation_overlap`). When overlaps sent
  the vocabulary back to be redrafted, the re-check asks at the backed-off
  threshold (§11.5), so the loop settles.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from jev_graph_builder.ids import pairs, sha256_hex, short_key
from jev_graph_builder.jev.gating import ACCEPT, REJECT
from jev_graph_builder.jev.service import Decision, JevService

QS_SUPPORT, QS_OVERLAP, QS_RELATION_SUPPORT = "ontology_support", "relation_overlap", "relation_support"
RELATIONS = "relation_types"


def cited_pairs(item: dict[str, Any], passages: dict[str, Any]) -> list[dict[str, str]]:
    """A relation's evidence as {source, target} texts. An entry is either a {source, target} pair of
    passage ids, or the id of a passage that is itself a (source, target) pair (§8.7.4 examples)."""
    out = []
    for e in item.get("evidence") or []:
        if isinstance(e, dict):
            src, dst = passages.get(e.get("source")), passages.get(e.get("target"))
            if isinstance(src, str) and isinstance(dst, str):
                out.append({"source": src, "target": dst})
        elif isinstance(passages.get(e), tuple):
            src, dst = passages[e]
            out.append({"source": src, "target": dst})
    return out


@dataclass
class CheckReport:
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    overlaps: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"unsupported": self.unsupported, "overlaps": self.overlaps,
                "decision_ids": [d.decision_id for d in self.decisions]}


async def check_definitions(jev: JevService, kind: str, items: list[dict[str, Any]], passages: dict[str, Any],
                            concurrency: int, only: set[str] | None = None,
                            supported: Callable[[int, int], bool] | None = None,
                            overlap_repeat: int = 0) -> CheckReport:
    """`passages` maps a citation ID to sample text, or to a (source, target) pair of texts;
    items cite them in `evidence` (see `cited_pairs` for relation types).

    With `only` (e.g. new proposals), support is checked for those items and
    overlap for the pairs that include at least one of them; otherwise for all.
    A relation type is supported when `supported(yes, total)` holds for Jev's answers on its cited
    pairs; by default, when Jev confirms any of them.
    `overlap_repeat`: how many earlier checks of this vocabulary Jev flagged overlaps in; the
    overlap question's back-off repeat (§11.5). It is per question, not per pair, so a pair the
    redraft renamed or added is held to the same raised bar.
    """
    report = CheckReport()
    sem = asyncio.Semaphore(concurrency)

    def any_yes(yes: int, _total: int) -> bool:
        return yes > 0

    def record(item: dict[str, Any], decisions: list[Decision], rule: Callable[[int, int], bool] = any_yes) -> None:
        report.decisions.extend(decisions)
        entry = {"kind": kind, "name": item["name"], "decision_ids": [d.decision_id for d in decisions]}
        if rule(sum(d.outcome == ACCEPT for d in decisions), len(decisions)):
            return
        report.unsupported.append(entry)

    async def support(item: dict[str, Any]) -> None:
        cited = [passages[c] for c in item.get("evidence") or [] if isinstance(passages.get(c), str)]
        if not cited:  # nothing to judge against: unsupported without asking Jev
            report.unsupported.append({"kind": kind, "name": item["name"], "decision_ids": []})
            return
        async with sem:
            r = await jev.ask(QS_SUPPORT, {"definition": {"kind": kind, "name": item["name"], "definition": item["definition"]},
                                           "passages": cited}, "ontology_item", sha256_hex(kind, item["name"], item["definition"]))
        record(item, [r.single])

    async def relation_support(item: dict[str, Any]) -> None:
        cited = cited_pairs(item, passages)
        if not cited:
            report.unsupported.append({"kind": kind, "name": item["name"], "decision_ids": []})
            return
        definition = {k: item.get(k, "") for k in ("name", "definition", "direction_semantics")}
        decisions: list[Decision] = []
        for start in range(0, len(cited), per_call):
            group = {short_key(p["source"], p["target"]): p for p in cited[start: start + per_call]}
            async with sem:
                r = await jev.ask(QS_RELATION_SUPPORT, {"definition": definition}, "ontology_item",
                                  sha256_hex(kind, item["name"], item["definition"], *group), fanout_items=group)
            decisions.extend(r.decisions)
        record(item, decisions, supported or any_yes)

    async def overlap(a: dict[str, Any], b: dict[str, Any]) -> None:
        async with sem:
            r = await jev.ask(QS_OVERLAP, {"relation_a": {"name": a["name"], "definition": a["definition"]},
                                           "relation_b": {"name": b["name"], "definition": b["definition"]}},
                              "relation_pair", sha256_hex(a["name"], a["definition"], b["name"], b["definition"]),
                              repeat=overlap_repeat)
        d = r.single
        report.decisions.append(d)
        if d.outcome != REJECT:  # Jev says they overlap
            report.overlaps.append({"a": a["name"], "b": b["name"], "outcome": d.outcome, "decision_id": d.decision_id})

    def wanted(*xs: dict[str, Any]) -> bool:
        return only is None or any(x["name"] in only for x in xs)

    per_call = jev.reg.policy("bootstrap.relation_pairs_per_call")
    tasks = [(relation_support if kind == RELATIONS else support)(i) for i in items if wanted(i)]
    if kind == RELATIONS:
        tasks.extend(overlap(a, b) for a, b in pairs(items) if wanted(a, b))
    await asyncio.gather(*tasks)
    return report
