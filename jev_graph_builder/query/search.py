"""Semantic search (§9.1): ANN + full-text → RRF fusion (code) → Jev rerank.

No harness call is made. Candidates the injection Noul flags are dropped.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.jev.gating import says_yes
from jev_graph_builder.jev.service import Decision
from jev_graph_builder.pipeline.common import Context, write_decisions

QS_PASSAGE = "rag_passage"
SUBJECT = "query_passage"


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    source_uri: str
    text: str
    title: str | None
    heading_path: list[str]
    char_start: int | None
    char_end: int | None
    rrf: float
    score: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    decision_id: str | None = None
    injection: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def rrf_fuse(rankings: list[list[str]], k: float) -> dict[str, float]:
    """Reciprocal rank fusion: Σ 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)
    return scores


def composite(answers: dict[str, dict[str, Any]], weights: dict[str, float]) -> float:
    return float(sum(w * answers[q]["p"] for q, w in weights.items() if q in answers))


async def candidates(ctx: Context, query: str, filters: dict[str, Any] | None = None) -> list[Hit]:
    pol = ctx.reg.policy("search")
    filters = filters or {}
    vec = np.asarray((await ctx.embedder.embed([query]))[0], dtype=np.float32)
    where = "c.corpus_id = %(c)s AND c.status = 'accepted' AND NOT coalesce(d.quarantined, false)"
    params: dict[str, Any] = {"c": ctx.corpus_id, "v": vec, "q": query, "k_ann": pol["k_ann"], "k_fts": pol["k_fts"]}
    if filters.get("doc_type"):
        where += " AND d.doc_type = ANY(%(dt)s)"
        params["dt"] = list(filters["doc_type"])
    if filters.get("role"):
        where += " AND c.role = ANY(%(role)s)"
        params["role"] = list(filters["role"])
    if filters.get("doc_id"):
        where += " AND c.doc_id = ANY(%(doc)s)"
        params["doc"] = list(filters["doc_id"])
    async with ctx.db.tx() as conn:
        await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(pol["ef_search"]),))
        cur = await conn.execute(
            f"SELECT chunk_id FROM (SELECT c.chunk_id, c.embedding <=> %(v)s AS dist FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE {where} AND c.embedding IS NOT NULL ORDER BY c.embedding <=> %(v)s LIMIT %(fetch)s) s "
            "ORDER BY dist, chunk_id LIMIT %(k_ann)s", {**params, "fetch": params["k_ann"] * ctx.reg.policy("store.ann_overfetch")})
        ann = [r["chunk_id"] for r in await cur.fetchall()]
    fts = [r["chunk_id"] for r in await ctx.db.fetch(
        f"SELECT c.chunk_id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE {where} "
        "AND c.fts @@ websearch_to_tsquery('simple', %(q)s) ORDER BY ts_rank(c.fts, websearch_to_tsquery('simple', %(q)s)) DESC, c.chunk_id "
        "LIMIT %(k_fts)s", params)]
    fused = rrf_fuse([ann, fts], pol["rrf_k"])
    top = sorted(fused, key=lambda c: (-fused[c], c))[: pol["rerank_candidates"]]
    rows = {r["chunk_id"]: r for r in await ctx.db.fetch(
        "SELECT c.chunk_id, c.doc_id, d.source_uri, c.text, c.title, c.heading_path, u1.char_start, u2.char_end "
        "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        "LEFT JOIN units u1 ON u1.unit_id = c.unit_ids[1] LEFT JOIN units u2 ON u2.unit_id = c.unit_ids[cardinality(c.unit_ids)] "
        "WHERE c.chunk_id = ANY(%s)", (top,))}
    return [Hit(chunk_id=c, doc_id=rows[c]["doc_id"], source_uri=rows[c]["source_uri"], text=rows[c]["text"], title=rows[c]["title"],
                heading_path=rows[c]["heading_path"] or [], char_start=rows[c]["char_start"], char_end=rows[c]["char_end"], rrf=fused[c])
            for c in top if c in rows]


async def rerank(ctx: Context, query: str, hits: list[Hit]) -> tuple[list[Hit], list[Decision]]:
    """Jev `QS.rag_passage` per candidate, concurrently; drop injection-flagged passages."""
    pol = ctx.reg.policy("search")
    sem = asyncio.Semaphore(ctx.reg.policy("run.concurrency.search"))
    q_inj = pol["injection_question"]

    by_rrf = sorted(hits, key=lambda h: (-h.rrf, h.chunk_id))

    async def one(h: Hit) -> Decision:
        # `contradicts_other_evidence` (§9.1): the passage is judged against the other top candidates.
        peers = [p.text for p in by_rrf if p.chunk_id != h.chunk_id][: pol["contradiction_peers"]]
        inputs = {"query": query, "passage": h.text, "other_evidence": peers or None}
        only = None if peers else set(ctx.reg.question_set(QS_PASSAGE).questions) - {pol["contradiction_question"]}
        async with sem:
            r = await ctx.jev.ask(QS_PASSAGE, inputs, SUBJECT, sha256_hex(query, h.chunk_id, sorted(p.chunk_id for p in by_rrf)), only=only)
        return r.single

    decisions = await asyncio.gather(*(one(h) for h in hits))
    kept = []
    for h, d in zip(hits, decisions, strict=True):
        h.decision_id = d.decision_id
        h.probabilities = {q: a["p"] for q, a in d.answers.items() if a["type"] == "noul"}
        h.injection = says_yes(d.answers[q_inj], d.threshold_for(q_inj))
        h.score = composite(d.answers, pol["weights"])
        if h.injection:
            continue
        kept.append(h)
    kept.sort(key=lambda h: (-h.score, -h.rrf, h.chunk_id))
    return kept, list(decisions)


async def search(ctx: Context, query: str, k: int | None = None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    hits = await candidates(ctx, query, filters)
    ranked, decisions = await rerank(ctx, query, hits)
    await persist(ctx, decisions)
    k = k or ctx.reg.policy("search.k")
    return {"query": query, "results": [h.as_dict() for h in ranked[:k]], "decision_ids": [d.decision_id for d in decisions]}


async def persist(ctx: Context, decisions: list[Decision]) -> None:
    """Query-time decisions are audited like pipeline ones (P3)."""
    if decisions:
        async with ctx.db.tx() as conn:
            await write_decisions(conn, *decisions)
