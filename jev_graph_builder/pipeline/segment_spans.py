"""Chunk spans from Jev's decisions (§8.3). Pure and deterministic.

A gap is a chunk boundary exactly where Jev said the two sides do not belong
together (`runs`). A run Jev kept together that is longer than one chunk can
hold is split at the gap Jev was least sure belongs together (`weakest_gap`),
among the gaps whose left side fits in one chunk (`split_gaps`).
"""

from __future__ import annotations


def runs(n: int, boundary: list[bool]) -> list[tuple[int, int]]:
    """Units [0, n) as half-open ranges between Jev's boundaries. `boundary[g]` is Jev's cut after unit g."""
    if len(boundary) != max(n - 1, 0):
        raise ValueError("need one boundary decision per gap")
    out, start = [], 0
    for g, cut in enumerate(boundary):
        if cut:
            out.append((start, g + 1))
            start = g + 1
    if n:
        out.append((start, n))
    return out


def split_gaps(tokens: list[int], start: int, end: int, max_tokens: int) -> list[int]:
    """Gaps g (cut after unit g) inside [start, end) whose left side [start, g] fits in `max_tokens`.

    Always non-empty for a run of two or more units: the first gap is allowed even
    when a single unit exceeds the cap, so every split makes progress.
    """
    out, size = [], 0
    for g in range(start, end - 1):
        size += tokens[g]
        if size > max_tokens and out:
            break
        out.append(g)
        if size > max_tokens:
            break
    return out


def weakest_gap(gaps: list[int], keep: list[float]) -> int:
    """The gap Jev was least sure keeps its two sides together (`keep[g]`); a tie goes to the later
    gap, so the left chunk is as large as the cap allows."""
    return min(reversed(gaps), key=lambda g: keep[g])
