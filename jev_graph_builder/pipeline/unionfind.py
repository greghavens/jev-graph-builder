"""Union-find for §8.6 entity merges (pure, deterministic).

Jev decides every pair whose names differ; entities of one type with the same normalized
name are one entity by code (`identical`). Code applies that "same thing" is transitive.
A merge of two clusters is refused when Jev said "not the same" for any pair across them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MergeOutcome:
    parent: dict[str, str]
    accepted: list[tuple[str, str]] = field(default_factory=list)
    refused: list[tuple[str, str, str]] = field(default_factory=list)  # (a, b, reason)

    def clusters(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for node in self.parent:
            out.setdefault(self.find(node), []).append(node)
        return {r: sorted(m) for r, m in out.items()}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root


JEV_SAID_DIFFERENT = "jev_said_different"


def merge(
    nodes: list[str],
    pairs: list[tuple[str, str, float]],
    known: dict[frozenset[str], bool],
    identical: list[tuple[str, str]] = (),
) -> MergeOutcome:
    """Merge `identical` pairs first, then the pairs Jev accepted, most confident first; refuse a
    merge Jev contradicted.

    `known` holds Jev's decision (accepted or not) for every pair it judged, so a
    transitive merge is refused when Jev already said two of its members differ.
    Roots are the lexicographically smallest member, so the result is deterministic.
    """
    out = MergeOutcome(parent={n: n for n in nodes})
    members: dict[str, list[str]] = {n: [n] for n in nodes}
    ordered = [*sorted(identical), *((a, b) for a, b, _p in sorted(pairs, key=lambda t: (-t[-1], t[0], t[1])))]
    for a, b in ordered:
        ra, rb = out.find(a), out.find(b)
        if ra == rb:
            continue
        ma, mb = members[ra], members[rb]
        if any(known.get(frozenset((x, y))) is False for x in ma for y in mb):
            out.refused.append((a, b, JEV_SAID_DIFFERENT))
            continue
        root, child = (ra, rb) if ra < rb else (rb, ra)
        out.parent[child] = root
        members[root] = sorted(members[root] + members.pop(child))
        out.accepted.append((a, b))
    for n in nodes:
        out.find(n)
    return out
