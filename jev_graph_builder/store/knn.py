"""k-nearest-neighbour reads for pipeline candidate generation (§8.6, §8.7.1).

Candidate sets feed Jev decisions, so they must be identical on every run over
the same data, including a run resumed after a crash (§19 acceptance 2). Three
modes, chosen by `policies.store.candidate_knn.mode`:

- `exact`: index scans are disabled for the one transaction, so pgvector scans
  and sorts by `(distance, id)`. The result is independent of index build order
  and of how equidistant rows sit on disk. Work grows with rows × rows.
- `hnsw`: the approximate index answers, with `hnsw.ef_search` raised to at
  least the rows requested. Fast at scale, but when vectors tie at the cut the
  chosen neighbours depend on how the index graph was built, so a resumed run
  can pick different candidates.
- `torch`: exact, like `exact`, but the all-pairs self-joins (`self_join`) run as
  matrix products on `candidate_knn.device` (the GPU when there is one) instead of
  one Postgres backend. The GPU only shortlists; the final order is float64
  `(distance, id)` on the CPU, and a row whose cut the shortlist cannot prove is
  ranked over every row, so the result does not depend on the device. Distances
  are float64 where pgvector's are float32, so rows closer than float32 noise at
  the cut can order differently from `exact`. Single-query reads run as `exact`.

The neighbour subquery must select the row id as its first column and the
distance as `dist`; its ORDER BY comes from `order_by`, which takes the
index-usable clause (`ORDER BY <distance expression>`) for `hnsw` mode.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from jev_graph_builder.store.db import Database

EXACT, HNSW, TORCH = "exact", "hnsw", "torch"
# Float32 dot products of unit vectors are within this of the float64 value (dim ≤ ~10k).
_F32_SLACK = 1e-4
_BLOCK_ELEMENTS = 1 << 28  # similarity block size: 1 GiB of float32


def order_by(policy: dict[str, Any], index_clause: str) -> str:
    return index_clause if policy["mode"] == HNSW else "ORDER BY dist, 1"


async def fetch(db: Database, policy: dict[str, Any], sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    async with db.tx() as conn:
        if policy["mode"] in (EXACT, TORCH):
            await conn.execute("SET LOCAL enable_indexscan = off")
        elif policy["mode"] == HNSW:
            width = max(int(policy["ef_search"]), int(params.get("fetch") or 0))
            await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(width),))
        else:
            raise ValueError(f"unknown candidate_knn mode {policy['mode']!r}")
        cur = await conn.execute(sql, params)
        return list(await cur.fetchall())


async def self_join(db: Database, pool_sql: str, params: dict[str, Any], k: int, fetch_n: int,
                    device: str = "auto") -> list[dict[str, Any]]:
    """`torch` mode: each pool row's k nearest other rows by cosine distance, ties by id.

    `pool_sql` selects `id` and `embedding`, and optionally `grp`: rows sharing a
    non-null `grp` are not paired. Returns `{"a": id, "b": neighbour id}` rows.
    """
    rows = await db.fetch(pool_sql, params)
    rows.sort(key=lambda r: r["id"])  # index order is id order, so index ties break by id
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    vecs = np.asarray([r["embedding"].to_numpy() for r in rows], dtype=np.float64)
    grp = None
    if "grp" in rows[0]:
        codes: dict[Any, int] = {}
        grp = np.asarray([-1 - i if r["grp"] is None else codes.setdefault(r["grp"], len(codes))
                          for i, r in enumerate(rows)])
    nbrs = await asyncio.to_thread(_neighbours, vecs, grp, k, fetch_n, device)
    return [{"a": ids[i], "b": ids[j]} for i, row in enumerate(nbrs) for j in row]


def _neighbours(vecs: np.ndarray, grp: np.ndarray | None, k: int, fetch_n: int, device: str) -> list[list[int]]:
    """Row i's k nearest allowed rows, ordered by float64 (distance, index).

    Each row's float32 shortlist is ranked in float64. A row whose k-th pick is not clearly closer
    than the shortlist's last float32 score (a tie cluster wider than the shortlist) goes round
    again with a shortlist eight times wider, until the cut is proven or the shortlist is every row.
    """
    import torch

    norms = np.linalg.norm(vecs, axis=1)
    live = norms > 0  # pgvector's cosine distance to a zero vector is NaN: never a neighbour
    x = np.divide(vecs, norms[:, None], out=np.zeros_like(vecs), where=live[:, None])
    n, dim = x.shape
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    x64 = torch.from_numpy(x).to(dev)
    x32 = x64.float()
    dead = torch.from_numpy(~live).to(dev)
    gg = torch.from_numpy(grp).to(dev) if grp is not None else None
    out: list[list[int]] = [[] for _ in range(n)]
    todo = torch.from_numpy(np.flatnonzero(live)).to(dev)
    m = min(max(fetch_n, k), n)
    while todo.numel():
        block = max(1, min(_BLOCK_ELEMENTS // n, _BLOCK_ELEMENTS // (2 * m * dim)))
        again = []
        for s in range(0, todo.numel(), block):
            rows = todo[s:s + block]
            sim = x32[rows] @ x32.T
            banned = dead[None, :].expand(len(rows), n).clone()
            banned[torch.arange(len(rows), device=dev), rows] = True
            if gg is not None:
                banned |= gg[rows, None] == gg[None, :]
            sim.masked_fill_(banned, -torch.inf)
            vals, cand = sim.topk(m, dim=1)
            del sim, banned
            d = 1.0 - (x64[cand] * x64[rows, None, :]).sum(-1)
            d.masked_fill_(torch.isinf(vals), torch.inf)
            cand, p = cand.sort(dim=1)  # ascending index, so the stable sort by distance breaks ties by index
            d, q = d.gather(1, p).sort(dim=1, stable=True)
            cand, d = cand.gather(1, q)[:, :k], d[:, :k]
            # Every row left out scored <= vals[:, -1] in float32: the cut is proven when nothing
            # allowed was left out or the k-th pick is clearly closer than that.
            complete = torch.isinf(vals[:, -1])
            proven = complete | (1.0 - d[:, -1] > vals[:, -1].double() + _F32_SLACK)
            for i, row, dist in zip(rows[proven].tolist(), cand[proven].tolist(), d[proven].tolist()):
                out[i] = [j for j, dj in zip(row, dist) if dj != float("inf")]
            again.append(rows[~proven])
        todo = torch.cat(again)
        m = min(m * 8, n)
    return out
