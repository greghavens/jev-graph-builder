"""S6: link discovery and validation — Jev determines the links (§8.7).

Candidates (code, policy-capped): ANN neighbours, structural neighbours,
shared resolved entities, full-text hits on key terms; pairs are unordered and
remember their sources. A pair belongs to the smaller chunk ID (its anchor);
each anchor is one ledger item, decided with `QS.link` per pair or
`QS.link_fanout` (anchor + candidates) by `policies.link.packing` (R-111).

Accepted semantic edges get an independent second opinion, `QS.link_verify`,
on claims and entities rather than raw text. Jev is the last model to decide:
if `link_verify` says no, there is no edge; no harness is consulted. Edge
weight is Σ w_i · feature_i with weights from policy. Every edge carries a
relation type from the ontology: Jev's `none` or `other` is no edge, and
`other` picks feed ontology evolution: during S6 on a significant `other`
excess, and a final pass in `finalize` (§8.7.4).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from jev_graph_builder.ids import sha256_hex, short_key
from jev_graph_builder.jev.gating import ACCEPT, REJECT
from jev_graph_builder.jev.service import AskResult, Decision
from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.pipeline.common import (
    Context, Deps, RunOptions, RunReport, Stage, Writer, write_decisions,
)
from jev_graph_builder.pipeline.evolution import maybe_evolve_ontology
from jev_graph_builder.store import knn, repo

QS_LINK, QS_LINK_FANOUT, QS_VERIFY = "link", "link_fanout", "link_verify"
PAIR_KIND = "chunk_pair"
ANCHOR_PREFIX = "anchor:"
SRC_ANN, SRC_STRUCT, SRC_ENTITY, SRC_FTS = "ann", "structural", "shared_entity", "fts"
DIR_A_TO_B, DIR_B_TO_A, DIR_SYMMETRIC = "a_to_b", "b_to_a", "symmetric"
DIRECTION_DIRECTED, DIRECTION_SYMMETRIC = "directed", "symmetric"   # ontology `direction` values (schema enum)


def relation_definitions(ctx: Context) -> dict[str, dict[str, Any]]:
    """Ontology relations plus the question set's added options (`other`, `none`)."""
    defs = {r["name"]: r for r in ctx.reg.ontology("relation_types")}
    crit = ctx.reg.question_set(QS_LINK).questions[ctx.reg.policy("link.relation_question")]["criteria"]
    for name, definition in (crit.get("add") or {}).items():
        defs.setdefault(name, {"name": name, "definition": definition, "direction": DIRECTION_SYMMETRIC})
    return defs


def edge_weight(features: dict[str, Any], weights: dict[str, float]) -> float:
    """§8.7.3: weight = Σ w_i · feature_i (booleans count as 0/1)."""
    return float(sum(w * float(features.get(k) or 0) for k, w in weights.items()))


class LinkStage(Stage):
    name = "links"
    deps = Deps(
        question_sets=(QS_LINK, QS_LINK_FANOUT, QS_VERIFY),
        policies=("link", "ontology", "ingest.metadata_fields"),
        ontology=("relation_types",),
        embedding=True,
    )

    # -------------------------------------------------------------- candidates

    async def candidates(self, ctx: Context) -> dict[tuple[str, str], set[str]]:
        caps = ctx.reg.policy("link.candidates")
        pairs: dict[tuple[str, str], set[str]] = defaultdict(set)

        def add(a: str, b: str, source: str) -> None:
            if a != b:
                pairs[(min(a, b), max(a, b))].add(source)

        knn_policy = ctx.reg.policy("store.candidate_knn")
        order = knn.order_by(knn_policy, "ORDER BY c2.embedding <=> c.embedding")
        rows = await knn.fetch(ctx.db, knn_policy,
            "SELECT c.chunk_id AS a, n.chunk_id AS b FROM chunks c CROSS JOIN LATERAL ("
            " SELECT s.chunk_id FROM (SELECT c2.chunk_id, c2.embedding <=> c.embedding AS dist FROM chunks c2"
            "  WHERE c2.corpus_id = c.corpus_id AND c2.chunk_id <> c.chunk_id"
            "  AND c2.status = 'accepted' AND c2.embedding IS NOT NULL AND NOT coalesce(c2.boilerplate, false)"
            f"  AND (NOT %(x)s OR c2.doc_id <> c.doc_id) {order} LIMIT %(fetch)s) s"
            " ORDER BY s.dist, s.chunk_id LIMIT %(k)s) n "
            "WHERE c.corpus_id = %(c)s AND c.status = 'accepted' AND c.embedding IS NOT NULL AND NOT coalesce(c.boilerplate, false)",
            {"k": caps["ann_k"], "fetch": caps["ann_k"] * ctx.reg.policy("store.ann_overfetch"),
             "x": caps["ann_exclude_same_doc"], "c": ctx.corpus_id})
        for r in rows:
            add(r["a"], r["b"], SRC_ANN)

        if caps["structural_enabled"]:
            labels = [ctx.reg.structural_label("next")]
            rows = await ctx.db.fetch(
                "SELECT src_id AS a, dst_id AS b FROM edges WHERE corpus_id = %s AND structural AND rel = ANY(%s) AND status = 'accepted'",
                (ctx.corpus_id, labels))
            for r in rows:
                add(r["a"], r["b"], SRC_STRUCT)
            section = ctx.reg.structural_label("in_section")
            rows = await ctx.db.fetch(
                "SELECT e1.src_id AS a, e2.src_id AS b FROM edges e1 JOIN edges e2 ON e1.dst_id = e2.dst_id AND e1.src_id < e2.src_id "
                "WHERE e1.corpus_id = %s AND e1.rel = %s AND e2.rel = %s AND e1.status = 'accepted' AND e2.status = 'accepted' "
                "ORDER BY a, b",
                (ctx.corpus_id, section, section))
            for r in rows[: caps["structural_section_pairs"]]:
                add(r["a"], r["b"], SRC_STRUCT)

        rows = await ctx.db.fetch(
            "WITH m AS (SELECT DISTINCT m.chunk_id, coalesce(e.merged_into, e.entity_id) AS ent FROM mentions m "
            " JOIN entities e ON e.entity_id = m.entity_id JOIN chunks c ON c.chunk_id = m.chunk_id "
            " WHERE c.corpus_id = %(c)s AND m.status = 'accepted' AND c.status = 'accepted'), "
            "p AS (SELECT m1.chunk_id AS a, m2.chunk_id AS b, count(*) AS shared, "
            " row_number() OVER (PARTITION BY m1.chunk_id ORDER BY count(*) DESC, m2.chunk_id) AS rn "
            " FROM m m1 JOIN m m2 ON m1.ent = m2.ent AND m1.chunk_id <> m2.chunk_id GROUP BY m1.chunk_id, m2.chunk_id) "
            "SELECT a, b FROM p WHERE rn <= %(k)s",
            {"c": ctx.corpus_id, "k": caps["shared_entity_k"]})
        for r in rows:
            add(r["a"], r["b"], SRC_ENTITY)

        rows = await ctx.db.fetch(
            "SELECT c.chunk_id AS a, n.chunk_id AS b FROM chunks c CROSS JOIN LATERAL ("
            " SELECT c2.chunk_id FROM chunks c2 WHERE c2.corpus_id = c.corpus_id AND c2.chunk_id <> c.chunk_id"
            " AND c2.status = 'accepted' AND c2.fts @@ websearch_to_tsquery('simple', array_to_string(c.keywords, ' or '))"
            " ORDER BY ts_rank(c2.fts, websearch_to_tsquery('simple', array_to_string(c.keywords, ' or '))) DESC, c2.chunk_id"
            " LIMIT %(k)s) n WHERE c.corpus_id = %(c)s AND c.status = 'accepted' AND cardinality(c.keywords) > 0",
            {"k": caps["fts_k"], "c": ctx.corpus_id})
        for r in rows:
            add(r["a"], r["b"], SRC_FTS)
        return pairs

    async def _chunk_views(self, ctx: Context, ids: list[str]) -> dict[str, dict[str, Any]]:
        sep = ctx.reg.policy("segment.heading_separator")
        rows = await ctx.db.fetch(
            "SELECT chunk_id, doc_id, heading_path, title, summary, text, meta FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
        out = {}
        for r in rows:
            path = sep.join([*(r["heading_path"] or []), *([r["title"]] if r["title"] else [])])
            out[r["chunk_id"]] = {"title_path": path, "summary": r["summary"] or "", "text": r["text"], "doc_id": r["doc_id"],
                                  "meta": r["meta"]}
        return out

    async def _entities_of(self, ctx: Context, ids: list[str]) -> dict[str, dict[str, str]]:
        rows = await ctx.db.fetch(
            "SELECT DISTINCT m.chunk_id, coalesce(r.entity_id, e.entity_id) AS ent, coalesce(r.canonical_name, e.canonical_name) AS name "
            "FROM mentions m JOIN entities e ON e.entity_id = m.entity_id LEFT JOIN entities r ON r.entity_id = e.merged_into "
            "WHERE m.chunk_id = ANY(%s) AND m.status = 'accepted'", (ids,))
        out: dict[str, dict[str, str]] = defaultdict(dict)
        for r in rows:
            out[r["chunk_id"]][r["ent"]] = r["name"]
        return out

    async def _claims_of(self, ctx: Context, ids: list[str]) -> dict[str, list[str]]:
        rows = await ctx.db.fetch("SELECT chunk_id, text FROM claims WHERE chunk_id = ANY(%s) AND status = 'accepted' ORDER BY claim_id", (ids,))
        out: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            out[r["chunk_id"]].append(r["text"])
        return out

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        pairs = await self.candidates(ctx)
        by_anchor: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        for (a, b), sources in sorted(pairs.items()):
            by_anchor[a].append((b, sorted(sources)))
        ids = sorted({x for p in pairs for x in p})
        views = await self._chunk_views(ctx, ids)
        ents = await self._entities_of(ctx, ids)
        items = []
        for anchor, cands in sorted(by_anchor.items()):
            payload = {"anchor": anchor, "candidates": cands}
            fingerprint = [views[anchor], sorted(ents[anchor]), [(b, s, views[b], sorted(ents[b])) for b, s in cands]]
            items.append(self.item(ctx, ANCHOR_PREFIX + anchor, fingerprint, payload=payload))
        ctx.cache["s6_views"], ctx.cache["s6_entities"] = views, ents
        return items

    # ---------------------------------------------------------------- deciding

    def _state(self, view: dict[str, Any]) -> dict[str, Any]:
        return {k: view[k] for k in ("title_path", "summary", "text", "meta")}

    async def _decide(self, ctx: Context, anchor: str, cands: list[tuple[str, list[str]]]
                      ) -> tuple[list[tuple[str, list[str], Decision, list[Decision]]], list[AskResult]]:
        """Per candidate: (b, sources, Jev's decision, earlier decisions on the pair), plus every Jev
        result asked, so each decision is written (P6)."""
        views, ents = ctx.cache["s6_views"], ctx.cache["s6_entities"]
        shared = {b: sorted(set(ents[anchor].values()) & set(ents[b].values())) for b, _ in cands}
        sem = asyncio.Semaphore(ctx.reg.policy("link.pair_concurrency"))

        async def one(b: str, s: list[str]):
            async with sem:
                r = await ctx.jev.ask(QS_LINK, {"chunk_a": self._state(views[anchor]), "chunk_b": self._state(views[b]),
                                                "computed": {"shared_entities": shared[b]}}, PAIR_KIND, f"{anchor}:{b}")
                return b, s, r.single, r

        if ctx.reg.policy("link.packing") != "anchor_fanout":
            singles = await asyncio.gather(*(one(b, s) for b, s in cands))
            return [(b, s, d, []) for b, s, d, _ in singles], [r for *_, r in singles]
        out = []
        asked: list[AskResult] = []
        per_call = ctx.reg.policy("link.fanout_per_call")
        for start in range(0, len(cands), per_call):
            group = cands[start: start + per_call]
            keys = {short_key(anchor, b): (b, s) for b, s in group}
            items = {k: {**self._state(views[b]), "shared_entities": shared[b]} for k, (b, _) in keys.items()}
            r = await ctx.jev.ask(QS_LINK_FANOUT, {"anchor": self._state(views[anchor])}, "anchor", anchor,
                                  fanout_items=items)
            asked.append(r)
            by_item = r.by_item()
            out.extend((b, s, by_item[k], []) for k, (b, s) in keys.items())
        return out, asked

    async def _verify(self, ctx: Context, src: str, dst: str, relation: dict[str, Any]) -> AskResult:
        ents, claims = ctx.cache["s6_entities"], ctx.cache.get("s6_claims", {})
        inputs = {
            "source": {"entities": sorted(ents[src].values()), "claims": claims.get(src, [])},
            "target": {"entities": sorted(ents[dst].values()), "claims": claims.get(dst, [])},
            "relation": {"name": relation["name"], "definition": relation["definition"]},
        }
        return await ctx.jev.ask(QS_VERIFY, inputs, "edge", sha256_hex(src, dst, relation["name"]))

    async def _resolve(self, ctx: Context, anchor: str, b: str, sources: list[str], d: Decision, earlier: list[Decision],
                       rels: dict[str, Any]) -> dict[str, Any]:
        """One candidate pair → edge row, plus the Jev results and decisions behind it."""
        reg = ctx.reg
        q_rel, q_dir = reg.policy("link.relation_question"), reg.policy("link.direction_question")
        rel_name = d.answers[q_rel]["choice"]
        direction = d.answers[q_dir]["choice"]
        relation = rels[rel_name]
        directed = relation.get("direction") == DIRECTION_DIRECTED
        src, dst = (b, anchor) if direction == DIR_B_TO_A else (anchor, b)
        decisions: list[Decision] = [*earlier, d]
        results: list[AskResult] = []
        status = "rejected"
        verify_answers: dict[str, Any] = {}
        outcome = d.outcome
        if directed and direction == DIR_SYMMETRIC:
            outcome = REJECT      # Jev gave a directed relation no direction: its answers name no edge to write
        if outcome == ACCEPT:
            v = await self._verify(ctx, src, dst, relation)
            results.append(v)
            decisions.append(v.single)
            verify_answers = v.single.answers
            if v.single.outcome == ACCEPT:
                status = "accepted"
        features = self._features(ctx, d, verify_answers, sources)
        contradiction = ctx.flag_decision(d, reg.policy("link.contradiction_question"))
        edge = {
            "edge_id": sha256_hex(src, dst, rel_name), "corpus_id": ctx.corpus_id, "src_kind": "chunk", "src_id": src,
            "dst_kind": "chunk", "dst_id": dst, "rel": rel_name, "directed": directed, "structural": False,
            "weight": edge_weight(features, reg.policy("link.weights")), "features": features, "status": status,
            "decision_ids": [x.decision_id for x in decisions], "registry_version": ctx.registry_version, "created_run_id": ctx.run_id,
        }
        extra = []
        if contradiction is True and rel_name != reg.policy("link.contradiction_relation"):
            crel = reg.policy("link.contradiction_relation")
            # Same verification bar as the pair's own edge: it inherits that edge's status and decisions.
            extra.append({**edge, "edge_id": sha256_hex(anchor, b, crel), "src_id": anchor, "dst_id": b, "rel": crel,
                          "directed": False})
        return {"edge": edge, "extra": extra, "results": results, "decisions": decisions}

    def _features(self, ctx: Context, d: Decision, verify: dict[str, Any], sources: list[str]) -> dict[str, Any]:
        reg = ctx.reg
        a = d.answers
        strength = a[reg.policy("link.strength_question")]
        feats: dict[str, Any] = {
            "p_related": a[reg.policy("link.related_question")]["p"],
            "relation_confidence": a[reg.policy("link.relation_question")].get("confidence"),
            "strength": strength["score"] / max(strength["levels"] - 1, 1) if strength.get("levels") else strength["score"],
            "p_contradiction": a[reg.policy("link.contradiction_question")]["p"],
            "sources": sources,
        }
        for q, v in verify.items():
            if v.get("type") == "noul":
                feats[f"verify_{q}"] = v["p"]
        for s in (SRC_ANN, SRC_STRUCT, SRC_ENTITY, SRC_FTS):
            feats[f"src_{s}"] = s in sources
        return feats

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        anchor = item.payload["anchor"]
        cands = item.payload["candidates"]
        if "s6_claims" not in ctx.cache:
            ctx.cache["s6_claims"] = {}
        ctx.cache["s6_claims"].update(await self._claims_of(ctx, [anchor, *[b for b, _ in cands]]))
        rels = relation_definitions(ctx)
        decided, asked = await self._decide(ctx, anchor, cands)
        resolved = await asyncio.gather(*(self._resolve(ctx, anchor, b, s, d, earlier, rels) for b, s, d, earlier in decided))

        async def write(conn) -> str:
            await write_decisions(conn, *asked)
            new_ids = []
            for res in resolved:
                await write_decisions(conn, *res["results"])
                for e in [res["edge"], *res["extra"]]:
                    await repo.upsert(conn, "edges", e, key=("edge_id",))
                    new_ids.append(e["edge_id"])
            others = [b for b, _ in cands]
            await conn.execute(
                "UPDATE edges SET status = 'superseded' WHERE NOT structural AND src_kind = 'chunk' AND dst_kind = 'chunk' "
                "AND ((src_id = %(a)s AND dst_id = ANY(%(o)s)) OR (dst_id = %(a)s AND src_id = ANY(%(o)s))) "
                "AND NOT (edge_id = ANY(%(k)s)) AND status <> 'superseded'",
                {"a": anchor, "o": others, "k": new_ids})
            return DONE

        return write

    async def after_item(self, ctx: Context, report: RunReport) -> None:
        # One evolution attempt at a time; picks written meanwhile are tested on the next item.
        lock = self._lock(ctx)
        if lock.locked():
            return
        async with lock:
            await self._evolve(ctx, report, final=False)

    async def finalize(self, ctx: Context, report: RunReport, opts: RunOptions) -> None:
        async with self._lock(ctx):  # waits for an attempt still running from the last item
            await self._evolve(ctx, report, final=True)

    @staticmethod
    def _lock(ctx: Context) -> asyncio.Lock:
        return ctx.cache.setdefault("s6_evolution_lock", asyncio.Lock())

    async def _evolve(self, ctx: Context, report: RunReport, final: bool) -> None:
        version = await maybe_evolve_ontology(ctx, self, final=final)
        if version:
            report.proposals.append(version)
