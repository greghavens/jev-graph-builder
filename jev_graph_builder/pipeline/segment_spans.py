"""Chunk spans (§8.3). Pure and deterministic; no Jev call.

A heading that follows content starts a section (`starts_section`). A section
that fits the chunk cap is one chunk. A longer section is split at unit
boundaries into the fewest pieces that fit (`split_section`): each cut is the
fitting gap nearest an even share of the section, skipping a gap right after a
heading or after a lead-in that ends in a colon while another fitting gap exists.
"""

from __future__ import annotations

import math
from typing import Any

from jev_graph_builder.parse.parsers import HEADING


def runs(n: int, boundary: list[bool]) -> list[tuple[int, int]]:
    """Units [0, n) as half-open ranges between boundaries. `boundary[g]` is a cut after unit g."""
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


def starts_section(units: list[dict[str, Any]], g: int) -> bool:
    """Gap g (between units g and g+1) is a section start: a heading that follows non-heading content."""
    return units[g + 1]["kind"] == HEADING and units[g]["kind"] != HEADING


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


def leads_into_next(unit: dict[str, Any]) -> bool:
    """A unit that introduces the one after it: a heading, or a lead-in such as "Run this command:"."""
    return unit["kind"] == HEADING or unit["text"].rstrip().endswith(":")


def split_section(units: list[dict[str, Any]], start: int, end: int, max_tokens: int) -> list[tuple[int, int]]:
    """Units [start, end) as pieces that each fit `max_tokens` (a single unit over the cap stays whole)."""
    tokens = [u["tokens"] for u in units]
    out: list[tuple[int, int]] = []
    while end - start > 1 and sum(tokens[start:end]) > max_tokens:
        total = sum(tokens[start:end])
        target = total / math.ceil(total / max_tokens)
        gaps = split_gaps(tokens, start, end, max_tokens)
        clean = [g for g in gaps if not leads_into_next(units[g])] or gaps
        # Left-side size at each gap; a tie goes to the later gap, so the left piece is the larger.
        cut = min(reversed(clean), key=lambda g: abs(sum(tokens[start:g + 1]) - target))
        out.append((start, cut + 1))
        start = cut + 1
    out.append((start, end))
    return out


def spans(units: list[dict[str, Any]], max_tokens: int) -> list[tuple[int, int]]:
    """A document's chunks: its sections, each split to fit `max_tokens`."""
    n = len(units)
    sections = runs(n, [starts_section(units, g) for g in range(n - 1)])
    return [piece for s, e in sections for piece in split_section(units, s, e, max_tokens)]
