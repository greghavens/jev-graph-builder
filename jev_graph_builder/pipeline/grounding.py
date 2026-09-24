"""Span grounding (§8.4 step 2): every extracted span must occur in the chunk.

Exact match, then normalized match (casefold, whitespace and the Registry's
character folds), then fuzzy match at the policy threshold. When a span occurs
more than once the caller asks Jev to pick among the candidate lines; when it
does not occur at all the extraction is `ungrounded`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

EXACT, NORMALIZED, FUZZY, LOCATED, UNGROUNDED = "exact", "normalized", "fuzzy", "located", "ungrounded"


@dataclass
class Grounding:
    kind: str
    start: int | None = None
    end: int | None = None
    candidates: list[tuple[int, int]] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


def _normalize(text: str, folds: dict[str, str]) -> tuple[str, list[int]]:
    """Normalized text plus, for each normalized char, its index in the original."""
    out: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        ch = folds.get(ch, ch)
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" ")
            index.append(i)
            prev_space = True
            continue
        for c in ch.casefold():
            out.append(c)
            index.append(i)
        prev_space = False
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def _all(haystack: str, needle: str) -> list[int]:
    found, i = [], haystack.find(needle)
    while i != -1:
        found.append(i)
        i = haystack.find(needle, i + 1)
    return found


def ground(span: str, text: str, fuzzy_threshold: float, folds: dict[str, str]) -> Grounding:
    span = span.strip()
    if not span:
        return Grounding(UNGROUNDED)
    hits = _all(text, span)
    if hits:
        cands = [(h, h + len(span)) for h in hits]
        return Grounding(EXACT, *cands[0], candidates=cands)

    n_text, idx = _normalize(text, folds)
    n_span, _ = _normalize(span, folds)
    hits = _all(n_text, n_span) if n_span else []
    if hits:
        cands = [(idx[h], idx[h + len(n_span) - 1] + 1) for h in hits]
        return Grounding(NORMALIZED, *cands[0], candidates=cands)

    if n_span and n_text:
        al = fuzz.partial_ratio_alignment(n_span, n_text)
        if al is not None and al.score >= fuzzy_threshold and al.dest_end > al.dest_start:
            cand = (idx[al.dest_start], idx[al.dest_end - 1] + 1)
            return Grounding(FUZZY, *cand, candidates=[cand])
    return Grounding(UNGROUNDED)


def line_of(text: str, start: int, end: int) -> str:
    """The full line(s) around a candidate: what Jev sees when picking among candidates."""
    s = text.rfind("\n", 0, start) + 1
    e = text.find("\n", end)
    return text[s: e if e != -1 else len(text)]
