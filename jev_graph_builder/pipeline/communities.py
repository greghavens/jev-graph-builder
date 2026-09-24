"""Leiden community detection with guaranteed termination (§8.8).

`leidenalg`'s node-moving loop can oscillate forever on floating-point ties
(`n_iterations` bounds only the outer loop). Two mechanical guards, both
parameterised by `policies.audit.communities`:

- edge weights are quantised to `weight_decimals`, which removes near-ties and
  keeps the partition deterministic for identical inputs;
- the optimiser runs in a spawned process that is killed after `timeout_s`.
  On timeout the caller gets `None` and skips communities for this run; the
  next run retries them.
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing as mp
from typing import Any

import igraph as ig
import leidenalg


def usable_edges(edges: list[dict[str, Any]], decimals: int) -> list[tuple[str, str, float]]:
    """Leiden's modularity null model divides by total weight: zero / NaN weights
    make it degenerate, and an unweighted edge carries no community signal anyway."""
    out = []
    for e in edges:
        w = e["weight"]
        if w is None or not math.isfinite(w):
            continue
        w = round(float(w), decimals)
        if w > 0:
            out.append((e["src_id"], e["dst_id"], w))
    return out


def leiden(edges: list[tuple[str, str, float]], pol: dict[str, Any]) -> list[list[str]]:
    nodes = sorted({n for s, d, _ in edges for n in (s, d)})
    if not nodes:
        return []
    index = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(index[s], index[d]) for s, d, _ in edges])
    g.es["weight"] = [w for _, _, w in edges]
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=pol["resolution"], seed=pol["seed"],
                                    n_iterations=pol["iterations"])
    groups = [sorted(nodes[i] for i in members) for members in part]
    return sorted((m for m in groups if len(m) >= pol["min_size"]), key=lambda m: m[0])


def _worker(edges: list[tuple[str, str, float]], pol: dict[str, Any], conn: Any) -> None:
    conn.send(leiden(edges, pol))
    conn.close()


async def partition(edges: list[dict[str, Any]], pol: dict[str, Any]) -> list[list[str]] | None:
    """Communities of the weighted edge list, or None if the optimiser timed out."""
    usable = usable_edges(edges, pol["weight_decimals"])
    if not usable:
        return []
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(usable, pol, send), daemon=True)
    proc.start()
    send.close()
    try:
        ready = await asyncio.to_thread(recv.poll, pol["timeout_s"])
        if not ready:
            return None
        return recv.recv()
    except EOFError:
        # The worker died without answering (e.g. crashed in the C extension).
        return None
    finally:
        if proc.is_alive():
            proc.kill()
        await asyncio.to_thread(proc.join)
        recv.close()
