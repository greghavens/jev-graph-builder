"""Sentence-level citation checks (§8.8 community summaries, §8.9 training
filters, §9.2 step 6). Jev answers a Choice (supports / contradicts /
says_nothing) per sentence against the passages it cites."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.jev.service import Decision, JevService

QS_CITATION = "citation_check"
SUBJECT = "sentence"


@dataclass
class CitationReport:
    kept: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def fail_fraction(self) -> float:
        total = len(self.kept) + len(self.failed)
        return len(self.failed) / total if total else 0.0


async def check_sentences(jev: JevService, sentences: list[dict[str, Any]], passages: dict[str, str], concurrency: int,
                          subject_prefix: str = "") -> CitationReport:
    """`sentences` are `{text, citation_ids}`; a sentence without a resolvable citation fails."""
    report = CitationReport()
    results: list[tuple[dict[str, Any], Decision | None]] = [None] * len(sentences)  # type: ignore[list-item]
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int, s: dict[str, Any]) -> None:
        cited = [passages[c] for c in s.get("citation_ids") or [] if c in passages]
        if not cited:
            results[i] = (s, None)
            return
        async with sem:
            r = await jev.ask(QS_CITATION, {"sentence": s["text"], "passages": cited}, SUBJECT,
                              sha256_hex(subject_prefix, s["text"], sorted(s.get("citation_ids") or [])))
        results[i] = (s, r.single)

    await asyncio.gather(*(one(i, s) for i, s in enumerate(sentences)))
    for s, d in results:
        if d is not None:
            report.decisions.append(d)
        entry = {**s, "decision_id": d.decision_id if d else None, "outcome": d.outcome if d else None}
        (report.kept if d is not None and d.outcome == ACCEPT else report.failed).append(entry)
    return report
