"""k-nearest-neighbour reads for pipeline candidate generation (§8.6, §8.7.1).

Candidate sets feed Jev decisions, so they must be identical on every run over
the same data, including a run resumed after a crash (§19 acceptance 2). Two
modes, chosen by `policies.store.candidate_knn.mode`:

- `exact`: index scans are disabled for the one transaction, so pgvector scans
  and sorts by `(distance, id)`. The result is independent of index build order
  and of how equidistant rows sit on disk. Work grows with rows × rows.
- `hnsw`: the approximate index answers, with `hnsw.ef_search` raised to at
  least the rows requested. Fast at scale, but when vectors tie at the cut the
  chosen neighbours depend on how the index graph was built, so a resumed run
  can pick different candidates.

The neighbour subquery must select the row id as its first column and the
distance as `dist`; its ORDER BY comes from `order_by`, which takes the
index-usable clause (`ORDER BY <distance expression>`) for `hnsw` mode.
"""

from __future__ import annotations

from typing import Any

from jev_graph_builder.store.db import Database

EXACT, HNSW = "exact", "hnsw"


def order_by(policy: dict[str, Any], index_clause: str) -> str:
    return "ORDER BY dist, 1" if policy["mode"] == EXACT else index_clause


async def fetch(db: Database, policy: dict[str, Any], sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    async with db.tx() as conn:
        if policy["mode"] == EXACT:
            await conn.execute("SET LOCAL enable_indexscan = off")
        elif policy["mode"] == HNSW:
            width = max(int(policy["ef_search"]), int(params.get("fetch") or 0))
            await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(width),))
        else:
            raise ValueError(f"unknown candidate_knn mode {policy['mode']!r}")
        cur = await conn.execute(sql, params)
        return list(await cur.fetchall())
