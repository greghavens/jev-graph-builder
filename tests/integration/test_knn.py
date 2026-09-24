"""Candidate kNN reproducibility (store/knn.py): neighbours must not depend on insertion order.

Many rows share identical vectors, so the `fetch` cut falls inside a tie. The exact mode
must return the same pairs whatever order the rows were written in, which is what a
resumed run needs (acceptance 2).
"""

from __future__ import annotations

import random

import pytest

from jev_graph_builder.store import knn
from jev_graph_builder.store.db import Database

pytestmark = pytest.mark.integration

ROWS, DIM, DISTINCT, K, FETCH = 600, 32, 12, 12, 48


def _vectors() -> dict[str, list[float]]:
    rng = random.Random(1)
    base = [[round(rng.gauss(0, 1), 3) for _ in range(DIM)] for _ in range(DISTINCT)]
    return {f"r{i:04d}": base[i % DISTINCT] for i in range(ROWS)}


async def _pairs(dsn: str, order: list[str], vecs: dict[str, list[float]]) -> set[tuple[str, str]]:
    db = Database(dsn, min_size=1, max_size=1)
    await db.open()
    try:
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await db.execute("DROP TABLE IF EXISTS knn_t")
        await db.execute(f"CREATE TABLE knn_t (id text PRIMARY KEY, v vector({DIM}))")
        await db.execute("CREATE INDEX ON knn_t USING hnsw (v vector_cosine_ops)")
        for key in order:
            await db.execute("INSERT INTO knn_t VALUES (%s, %s::vector)", (key, str(vecs[key])))
        policy = {"mode": knn.EXACT, "ef_search": 40}
        order_sql = knn.order_by(policy, "ORDER BY b.v <=> a.v")
        rows = await knn.fetch(db, policy,
            "SELECT a.id AS a, n.id AS b FROM knn_t a CROSS JOIN LATERAL (SELECT s.id FROM ("
            " SELECT b.id, b.v <=> a.v AS dist FROM knn_t b WHERE b.id <> a.id"
            f" {order_sql} LIMIT %(fetch)s) s ORDER BY s.dist, s.id LIMIT %(k)s) n",
            {"fetch": FETCH, "k": K})
        return {(r["a"], r["b"]) for r in rows}
    finally:
        await db.close()


async def test_exact_knn_ignores_insertion_order(dsn: str) -> None:
    vecs = _vectors()
    keys = sorted(vecs)
    forward = await _pairs(dsn, keys, vecs)
    backward = await _pairs(dsn, list(reversed(keys)), vecs)
    assert len(forward) == ROWS * K
    assert forward == backward
