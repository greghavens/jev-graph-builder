"""S7: graph audit — code finds, Jev judges (§8.8).

Code flags anomalies (orphan chunks, hub edges above a degree percentile,
contradiction edges, asymmetric pairs under symmetric relations, edges joining
unrelated doc types inside one component). Jev judges each flagged item with
`QS.audit_edge` / `QS.audit_orphan`; the result is a fix (demote the edge,
re-queue the pair) or a review item.

Finalize steps (ledger items too): drift/consistency sampling and Leiden
communities with harness summaries whose sentences Jev citation-checks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import igraph as ig
import numpy as np

from jev_graph_builder import log
from jev_graph_builder.calibrate.drift import DriftStats
from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.jev.gating import ACCEPT, REJECT
from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.pipeline.citations import check_sentences
from jev_graph_builder.pipeline.communities import partition as leiden_partition
from jev_graph_builder.pipeline.common import (
    Context, Deps, RunOptions, RunReport, Stage, Writer, run_single_item, write_decisions,
)
from jev_graph_builder.pipeline.s6_links import ANCHOR_PREFIX, relation_definitions
from jev_graph_builder.store import repo

QS_EDGE, QS_ORPHAN = "audit_edge", "audit_orphan"
PROMPT_COMMUNITY, SCHEMA_COMMUNITY = "community_summary", "community_summary"
EDGE_PREFIX, ORPHAN_PREFIX = "edge:", "orphan:"
HUB, CONTRADICTION, ASYMMETRIC, CROSS_DOCTYPE = "hub", "contradiction", "asymmetric", "cross_doctype"
LINK_STAGE = "links"
ALERT_COMMUNITIES = "communities_timeout"


class AuditStage(Stage):
    name = "audit"
    deps = Deps(question_sets=(QS_EDGE, QS_ORPHAN), prompts=(PROMPT_COMMUNITY,), schemas=(SCHEMA_COMMUNITY,),
                policies=("audit", "ingest.metadata_fields"), ontology=("relation_types",))

    # ------------------------------------------------------------- anomalies

    async def _graph(self, ctx: Context) -> tuple[list[dict[str, Any]], dict[str, str]]:
        edges = await ctx.db.fetch(
            "SELECT edge_id, src_id, dst_id, rel, directed, weight, status, decision_ids FROM edges "
            "WHERE corpus_id = %s AND NOT structural AND src_kind = 'chunk' AND dst_kind = 'chunk' AND status IN ('accepted', 'rejected')",
            (ctx.corpus_id,))
        doc_types = {r["chunk_id"]: r["doc_type"] or "" for r in await ctx.db.fetch(
            "SELECT c.chunk_id, d.doc_type FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE c.corpus_id = %s AND c.status = 'accepted'",
            (ctx.corpus_id,))}
        return edges, doc_types

    def anomalies(self, ctx: Context, edges: list[dict[str, Any]], doc_types: dict[str, str]) -> dict[str, set[str]]:
        """edge_id → reasons. Pure over the fetched rows (unit-tested)."""
        pol = ctx.reg.policy("audit")
        accepted = [e for e in edges if e["status"] == "accepted"]
        reasons: dict[str, set[str]] = defaultdict(set)

        degree: dict[str, int] = defaultdict(int)
        for e in accepted:
            degree[e["src_id"]] += 1
            degree[e["dst_id"]] += 1
        if degree:
            cut = float(np.percentile(list(degree.values()), pol["hub_percentile"]))
            hubs = {n for n, d in degree.items() if d > cut and d >= pol["hub_min_degree"]}
            for e in accepted:
                if e["src_id"] in hubs or e["dst_id"] in hubs:
                    reasons[e["edge_id"]].add(HUB)

        contradiction = ctx.reg.policy("link.contradiction_relation")
        for e in accepted:
            if e["rel"] == contradiction:
                reasons[e["edge_id"]].add(CONTRADICTION)

        by_key = {(e["src_id"], e["dst_id"], e["rel"]): e for e in edges}
        for e in accepted:
            if e["directed"]:
                continue
            twin = by_key.get((e["dst_id"], e["src_id"], e["rel"]))
            if twin is not None and twin["status"] == "rejected":
                reasons[e["edge_id"]].add(ASYMMETRIC)

        related = {frozenset(p) for p in pol["related_doc_types"]}
        nodes = sorted({n for e in accepted for n in (e["src_id"], e["dst_id"])})
        index = {n: i for i, n in enumerate(nodes)}
        g = ig.Graph(n=len(nodes), edges=[(index[e["src_id"]], index[e["dst_id"]]) for e in accepted])
        membership = g.connected_components().membership if nodes else []
        comp_types: dict[int, set[str]] = defaultdict(set)
        for n, c in zip(nodes, membership, strict=True):
            comp_types[c].add(doc_types.get(n, ""))
        for e in accepted:
            a, b = doc_types.get(e["src_id"], ""), doc_types.get(e["dst_id"], "")
            if a != b and len(comp_types[membership[index[e["src_id"]]]]) > 1 and frozenset((a, b)) not in related:
                reasons[e["edge_id"]].add(CROSS_DOCTYPE)
        return reasons

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        edges, doc_types = await self._graph(ctx)
        reasons = self.anomalies(ctx, edges, doc_types)
        by_id = {e["edge_id"]: e for e in edges}
        items = [self.item(ctx, f"{EDGE_PREFIX}{eid}", by_id[eid]["decision_ids"], sorted(r),
                           payload={"edge_id": eid, "reasons": sorted(r)}) for eid, r in sorted(reasons.items())]
        linked = {n for e in edges if e["status"] == "accepted" for n in (e["src_id"], e["dst_id"])}
        orphans = await ctx.db.fetch(
            "SELECT chunk_id, text FROM chunks WHERE corpus_id = %s AND status = 'accepted' AND NOT coalesce(boilerplate, false) ORDER BY chunk_id",
            (ctx.corpus_id,))
        items.extend(self.item(ctx, f"{ORPHAN_PREFIX}{r['chunk_id']}", sha256_hex(r["text"]), payload={"chunk_id": r["chunk_id"]})
                     for r in orphans if r["chunk_id"] not in linked)
        return items

    # --------------------------------------------------------------- judging

    async def _view(self, ctx: Context, chunk_id: str) -> dict[str, Any]:
        r = await ctx.db.fetchone("SELECT title, text, meta FROM chunks WHERE chunk_id = %s", (chunk_id,)) or {}
        return {"title": r.get("title") or "", "meta": r.get("meta"), "text": r.get("text") or ""}

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        if item.item_id.startswith(ORPHAN_PREFIX):
            return await self._orphan(ctx, item)
        edge = await ctx.db.fetchone("SELECT edge_id, src_id, dst_id, rel FROM edges WHERE edge_id = %s", (item.payload["edge_id"],))
        rel = relation_definitions(ctx).get(edge["rel"]) if edge else None
        if edge is None or rel is None:
            return lambda conn: _done()
        r = await ctx.jev.ask(QS_EDGE, {"relation": {"name": rel["name"], "definition": rel["definition"]},
                                        "source": await self._view(ctx, edge["src_id"]), "target": await self._view(ctx, edge["dst_id"])},
                              "edge", edge["edge_id"])
        d = r.single

        async def write(conn) -> str:
            # The audit decision links to the edge by subject_id; the edge's own decision_ids stay
            # the admission provenance (and part of this item's input hash).
            await write_decisions(conn, r)
            if d.outcome == REJECT:
                await conn.execute("UPDATE edges SET status = %s WHERE edge_id = %s", (ctx.reg.policy("audit.demoted_status"), edge["edge_id"]))
            return DONE

        return write

    async def _orphan(self, ctx: Context, item: WorkItem) -> Writer:
        chunk_id = item.payload["chunk_id"]
        r = await ctx.jev.ask(QS_ORPHAN, {"chunk": await self._view(ctx, chunk_id)}, "chunk", chunk_id)
        d = r.single

        async def write(conn) -> str:
            await write_decisions(conn, r)
            if d.outcome == ACCEPT:  # should connect: re-run its link candidates
                await conn.execute("UPDATE work_items SET status = 'pending' WHERE stage = %s AND item_id = %s",
                                   (LINK_STAGE, f"{ANCHOR_PREFIX}{chunk_id}"))
            return DONE

        return write

    # -------------------------------------------------------------- finalize

    async def finalize(self, ctx: Context, report: RunReport, opts: RunOptions) -> None:
        await self.drift(ctx)
        await self.communities(ctx)

    async def drift(self, ctx: Context) -> dict[str, Any] | None:
        pol = ctx.reg.policy("audit.drift")
        item = self.item(ctx, f"drift:{ctx.run_id}", ctx.run_id)
        out: dict[str, Any] = {}

        async def factory() -> Writer:
            rows = await ctx.db.fetch(
                "SELECT call_id, question_set, qs_version, state, state_hash, questions, keys, dynamic, answers, jev_model FROM jev_calls "
                "WHERE NOT drift AND provider = %s AND state IS NOT NULL AND questions IS NOT NULL "
                "ORDER BY md5(call_id || %s) LIMIT %s", (ctx.jev.provider.name, ctx.run_id or "", pol["sample_size"]))
            stats = DriftStats()
            for row in rows:
                raw = await ctx.jev.replay(row)
                stats.add(row["answers"], raw.answers, pol["flip_boundary"])
                if raw.model != ctx.jev.provider.model:
                    stats.model_mismatch.append(raw.model)
            breaches = stats.breaches(pol)
            out.update(stats.as_dict(), breaches=breaches, sampled=len(rows))

            async def write(conn) -> str:
                # Every sample is recorded (breaches may be empty) so the latest row is the
                # current drift state that `audit drift` reports (§13).
                await repo.upsert(conn, "alerts", {"alert_id": sha256_hex("drift", ctx.run_id or ""), "kind": "drift",
                                                   "subject": ctx.jev.provider.model, "payload": out, "run_id": ctx.run_id},
                                  key=("alert_id",))
                if breaches:
                    log.get().warning("drift_breach", breaches=breaches, model=ctx.jev.provider.model)
                return DONE

            return write

        await run_single_item(ctx, self, item, factory)
        return out or None

    async def communities(self, ctx: Context) -> list[str]:
        pol = ctx.reg.policy("audit.communities")
        edges = await ctx.db.fetch(
            "SELECT src_id, dst_id, weight FROM edges WHERE corpus_id = %s AND NOT structural AND status = 'accepted' "
            "AND src_kind = 'chunk' AND dst_kind = 'chunk' ORDER BY edge_id", (ctx.corpus_id,))
        groups = await leiden_partition(edges, pol)
        if groups is None:
            log.get().warning("communities_timeout", edges=len(edges), timeout_s=pol["timeout_s"])
            await ctx.db.execute(
                "INSERT INTO alerts (alert_id, kind, subject, payload, run_id) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (alert_id) DO NOTHING",
                (sha256_hex(ALERT_COMMUNITIES, ctx.run_id or ""), ALERT_COMMUNITIES, ctx.corpus_id,
                 repo.Jsonb({"edges": len(edges), "timeout_s": pol["timeout_s"]}), ctx.run_id))
            return []
        ids = []
        for members in groups:
            community_id = sha256_hex(ctx.corpus_id, members)
            ids.append(community_id)
            texts = {r["chunk_id"]: r["text"] for r in await ctx.db.fetch(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s) ORDER BY chunk_id", (members,))}
            item = self.item(ctx, f"community:{community_id}", members, sha256_hex(texts))

            async def factory(members=members, community_id=community_id, texts=texts) -> Writer:
                shown = members[: pol["max_members_in_prompt"]]
                cite = {f"c{i + 1}": m for i, m in enumerate(shown)}
                passages = {c: texts[m] for c, m in cite.items() if m in texts}
                profile = await ctx.jobs.choose_profile(PROMPT_COMMUNITY, {"records": list(passages.values())})
                ws = ctx.jobs.workspace_for(PROMPT_COMMUNITY, community_id)
                ws.mkdir(parents=True, exist_ok=True)
                result, _ = await ctx.jobs.run_single(PROMPT_COMMUNITY, SCHEMA_COMMUNITY, profile,
                                                      {"passages": [{"id": c, "text": t} for c, t in passages.items()]}, ws)
                sentences = (result.structured or {}).get("sentences", []) if result.ok else []
                check = await check_sentences(ctx.jev, sentences, passages, ctx.reg.policy("run.concurrency.citation_check"), community_id)
                summary = " ".join(s["text"] for s in check.kept)
                # Jev's sentence checks decide; a summary with too many failed sentences is not published.
                rejected = not result.ok or not check.kept or check.fail_fraction > pol["max_failed_fraction"]

                async def write(conn) -> str:
                    await write_decisions(conn, *check.decisions)
                    await repo.upsert(conn, "communities", {
                        "community_id": community_id, "corpus_id": ctx.corpus_id, "level": pol["level"], "member_ids": members,
                        "summary": summary, "status": "rejected" if rejected else "accepted",
                        "decision_ids": [d.decision_id for d in check.decisions],
                        "registry_version": ctx.registry_version, "created_run_id": ctx.run_id}, key=("community_id",))
                    return DONE

                return write

            await run_single_item(ctx, self, item, factory)
        async with ctx.db.tx() as conn:
            await conn.execute("UPDATE communities SET status = 'superseded' WHERE corpus_id = %s AND NOT (community_id = ANY(%s)) "
                               "AND status <> 'superseded'", (ctx.corpus_id, ids))
        return ids


async def _done() -> str:
    return DONE
