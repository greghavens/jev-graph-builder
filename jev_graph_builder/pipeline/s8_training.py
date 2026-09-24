"""S8: training-set generation (§8.9).

Templates are Registry data (`training/templates/*.yaml`); `corpus.yaml` lists
the enabled ones. Code dispatches on the template `kind`:

  qa_single, instruction   one chunk → harness example → Jev filter set + citation check
  qa_multihop              a sampled path of accepted edges → harness → per-chunk
                           necessity Noul ("answerable without chunk_i" must be false)
                           + per-hop citation checks
  graph_reasoning          an accepted edge + endpoints → harness → citation check
                           against the edge evidence
  triples                  accepted edges and claims, exported by code (already verified)
  retrieval_pairs          accepted qa_* questions + positive chunk + hard negatives
                           (ANN neighbours Jev judges do not answer the query)
  rerank_soft_labels       accepted qa_* questions × ANN candidates → Jev relevance
                           probabilities as soft labels

Generated templates run in the `training` stage; derived ones (they need
accepted qa_* examples) in `training_derived`. Dedup (embedding ANN, then a
Jev "same question" Noul) and deterministic document/component splits run in
finalize. Every stored example carries the decision IDs that admitted it
(R-090).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jev_graph_builder.ids import sha256_hex, unit_fraction
from jev_graph_builder.jev.gating import ACCEPT, REJECT
from jev_graph_builder.jev.service import AskResult, Decision
from jev_graph_builder.ledger.ledger import DONE, REVIEW, WorkItem
from jev_graph_builder.pipeline.citations import check_sentences
from jev_graph_builder.pipeline.common import (
    Context, Deps, RunOptions, RunReport, Stage, Writer, enqueue_review, pending_in, run_single_item, write_decisions,
)
from jev_graph_builder.pipeline.grounding import UNGROUNDED, ground
from jev_graph_builder.pipeline.s6_links import relation_definitions
from jev_graph_builder.store import knn, repo

GENERATED = ("qa_single", "instruction", "qa_multihop", "graph_reasoning", "triples")
DERIVED = ("retrieval_pairs", "rerank_soft_labels")
QA_KINDS = ("qa_single", "qa_multihop")
CACHE_KEY = "s8_records"
QS_DEDUP = "dedup"
ACCEPTED, REJECTED, IN_REVIEW, DUPLICATE = "accepted", "rejected", "review", "duplicate"


def enabled_templates(ctx: Context, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    names = ctx.cache.get("train_only") or ctx.reg.corpus.get("training_templates") or []
    out = []
    for name in names:
        tpl = ctx.reg.training_template(name)
        if tpl["kind"] in kinds:
            out.append(tpl)
    return out


def stratified(rows: list[dict[str, Any]], keys: list[str], quota: int, salt: str) -> list[dict[str, Any]]:
    """Round-robin over strata, each stratum ordered by a deterministic hash (P5)."""
    strata: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        strata[tuple(r.get(k) or "" for k in keys)].append(r)
    queues = [sorted(v, key=lambda r: unit_fraction(salt, r["source_key"])) for _, v in sorted(strata.items())]
    out: list[dict[str, Any]] = []
    while len(out) < quota and any(queues):
        for q in queues:
            if q and len(out) < quota:
                out.append(q.pop(0))
    return out


def assign_split(key: str, fractions: dict[str, float]) -> str:
    """Deterministic split by document or connected component (no leakage)."""
    x = unit_fraction("split", key)
    acc = 0.0
    names = list(fractions)
    for name in names:
        acc += fractions[name]
        if x < acc:
            return name
    return names[-1]


def component_keys(edges: list[tuple[str, str]], chunk_doc: dict[str, str]) -> dict[str, str]:
    """chunk → split key: its connected component (union of documents joined by
    accepted edges) so no train/test pair shares a document."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        da, db = chunk_doc.get(a), chunk_doc.get(b)
        if da and db:
            ra, rb = find(da), find(db)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {c: find(d) for c, d in chunk_doc.items()}


class TrainingStage(Stage):
    name = "training"
    kinds = GENERATED

    @property
    def deps(self) -> Deps:  # type: ignore[override]
        return Deps(question_sets=("train_qa", "train_instruction", "train_multihop", "train_negative", "rag_passage",
                                   "citation_check", QS_DEDUP),
                    prompts=("gen_qa_single", "gen_instruction", "gen_qa_multihop", "gen_graph_reasoning"),
                    policies=("training",), corpus=True, embedding=True)

    # ------------------------------------------------------------- sources

    async def _chunks(self, ctx: Context) -> list[dict[str, Any]]:
        return [dict(r, source_key=r["chunk_id"]) for r in await ctx.db.fetch(
            "SELECT c.chunk_id, c.doc_id, c.text, c.title, c.topic, d.doc_type FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE c.corpus_id = %s AND c.status = 'accepted' AND NOT coalesce(c.boilerplate, false) ORDER BY c.chunk_id",
            (ctx.corpus_id,))]

    async def _edges(self, ctx: Context) -> list[dict[str, Any]]:
        return await ctx.db.fetch(
            "SELECT edge_id, src_id, dst_id, rel, directed, decision_ids, weight FROM edges WHERE corpus_id = %s AND NOT structural "
            "AND status = 'accepted' AND src_kind = 'chunk' AND dst_kind = 'chunk' ORDER BY edge_id", (ctx.corpus_id,))

    def paths(self, edges: list[dict[str, Any]], min_hops: int, max_hops: int, limit: int, salt: str, max_expansions: int) -> list[list[dict[str, Any]]]:
        """Deterministic simple paths of `min_hops..max_hops` accepted edges."""
        adj: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in edges:
            adj[e["src_id"]].append(e)
            if not e["directed"]:
                adj[e["dst_id"]].append({**e, "src_id": e["dst_id"], "dst_id": e["src_id"]})
        for v in adj.values():
            v.sort(key=lambda e: unit_fraction(salt, e["edge_id"]))
        out: list[list[dict[str, Any]]] = []
        span = max_hops - min_hops + 1
        for start in sorted(adj, key=lambda n: unit_fraction(salt, n)):
            if len(out) >= limit:
                break
            # One path per start node; its hop count is drawn deterministically from min..max.
            target = min_hops + int(unit_fraction(salt, "hops", start) * span)
            stack: list[tuple[str, list[dict[str, Any]]]] = [(start, [])]
            expansions = 0
            while stack and expansions < max_expansions:
                expansions += 1
                node, path = stack.pop()
                if len(path) == target:
                    out.append(path)
                    break
                seen = {start, *(e["dst_id"] for e in path)}
                stack.extend((e["dst_id"], [*path, e]) for e in reversed(adj[node]) if e["dst_id"] not in seen)
        return out

    async def sources(self, ctx: Context, tpl: dict[str, Any]) -> list[dict[str, Any]]:
        kind, name = tpl["kind"], tpl["name"]
        pol = ctx.reg.policy("training")
        quota = pol["quotas"][name]
        strat = pol["stratify_by"]
        if kind in ("qa_single", "instruction"):
            return stratified(await self._chunks(ctx), strat, quota, name)
        edges = await self._edges(ctx)
        if kind == "qa_multihop":
            mh = pol["multihop"]
            paths = self.paths(edges, mh["min_hops"], mh["max_hops"], quota, name, mh["max_expansions"])
            return [{"source_key": sha256_hex([e["edge_id"] for e in p]), "path": p} for p in paths]
        if kind == "graph_reasoning":
            return stratified([dict(e, source_key=e["edge_id"]) for e in edges], ["rel"], quota, name)
        if kind == "triples":
            claims = await ctx.db.fetch(
                "SELECT cl.claim_id, cl.chunk_id, cl.text, cl.claim_type, cl.decision_ids FROM claims cl JOIN chunks c ON c.chunk_id = cl.chunk_id "
                "WHERE c.corpus_id = %s AND cl.status = 'accepted' ORDER BY cl.claim_id", (ctx.corpus_id,))
            return [*(dict(e, source_key=e["edge_id"], triple="edge") for e in edges),
                    *(dict(c, source_key=c["claim_id"], triple="claim") for c in claims)]
        if kind in DERIVED:
            rows = await ctx.db.fetch(
                "SELECT example_id, payload, source_chunk_ids, decision_ids FROM training_examples WHERE template = ANY(%s) "
                "AND status = %s ORDER BY example_id", (pol["derive_from"], ACCEPTED))
            return [dict(r, source_key=r["example_id"]) for r in rows][:quota]
        raise ValueError(f"unknown training template kind `{kind}`")

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        items = []
        for tpl in enabled_templates(ctx, self.kinds):
            for src in await self.sources(ctx, tpl):
                payload = {"template": tpl["name"], "kind": tpl["kind"], "source": src}
                items.append(self.item(ctx, f"{tpl['name']}:{src['source_key']}", tpl, src, payload=payload))
        return items

    # ------------------------------------------------------------- harness

    def record(self, ctx: Context, kind: str, src: dict[str, Any]) -> dict[str, Any]:
        if kind in ("qa_single", "instruction"):
            return {"id": src["source_key"], "text": src["text"], "title": src.get("title") or ""}
        if kind == "qa_multihop":
            return {"id": src["source_key"], "passages": [{"id": cid, "text": ""} for cid in self._path_chunks(src)]}
        rels = relation_definitions(ctx)
        return {"id": src["source_key"], "relation": {"name": src["rel"], "definition": rels.get(src["rel"], {}).get("definition", "")},
                "source_chunk": src["src_id"], "target_chunk": src["dst_id"]}

    @staticmethod
    def _path_chunks(src: dict[str, Any]) -> list[str]:
        path = src["path"]
        return [path[0]["src_id"], *(e["dst_id"] for e in path)]

    async def _texts(self, ctx: Context, ids: list[str]) -> dict[str, str]:
        return {r["chunk_id"]: r["text"] for r in await ctx.db.fetch("SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (ids,))}

    async def prepare(self, ctx: Context, items: list[WorkItem], enumerated: list[WorkItem]) -> None:
        cache: dict[str, tuple[dict[str, Any] | None, str, str, str | None]] = ctx.cache.setdefault(CACHE_KEY, {})
        by_tpl: dict[str, list[WorkItem]] = defaultdict(list)
        for i in enumerated:
            by_tpl[i.payload["template"]].append(i)
        sem = asyncio.Semaphore(ctx.reg.policy("run.concurrency.harness_jobs"))

        async def run(group: list[WorkItem], todo_items: list[WorkItem]) -> None:
            name = group[0].payload["template"]
            tpl = ctx.reg.training_template(name)
            if not tpl.get("prompt"):
                return
            # Jev routes on the template's whole set, so a resumed run gets the same route decision.
            records = [self.record(ctx, tpl["kind"], i.payload["source"]) for i in group]
            texts = await self._texts(ctx, sorted({p["id"] for r in records for p in r.get("passages", [])}
                                                  | {r[k] for r in records for k in ("source_chunk", "target_chunk") if k in r}))
            for r in records:
                for p in r.get("passages", []):
                    p["text"] = texts.get(p["id"], "")
                for k in ("source_chunk", "target_chunk"):
                    if k in r:
                        r[k] = {"id": r[k], "text": texts.get(r[k], "")}
            async with sem:
                profile = await ctx.jobs.choose_profile(name, {"records": records, "prompt": tpl["prompt"]})
                by_item = {i.item_id: r for i, r in zip(group, records, strict=True)}
                todo = [by_item[i.item_id] for i in todo_items]
                cap = ctx.reg.profile("harness", profile).get("max_batch_records") or len(todo)
                for start in range(0, len(todo), cap):
                    part = todo[start: start + cap]
                    outcome = await ctx.jobs.run_job(name, tpl["prompt"], part, profile, variables={"template": tpl})
                    for r in part:
                        run_id = outcome.harness_run_ids[-1] if outcome.harness_run_ids else None
                        cache[f"{name}:{r['id']}"] = (outcome.records.get(r["id"]), profile, outcome.missing.get(r["id"], ""), run_id)

        await asyncio.gather(*(run(b, todo) for b, todo in pending_in(list(by_tpl.values()), items)))

    # ------------------------------------------------------------- filters

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        p = item.payload
        tpl = ctx.reg.training_template(p["template"])
        handler = {
            "qa_single": self._single, "instruction": self._single, "qa_multihop": self._multihop,
            "graph_reasoning": self._graph_reasoning, "triples": self._triple,
            "retrieval_pairs": self._retrieval_pair, "rerank_soft_labels": self._rerank,
        }[p["kind"]]
        return await handler(ctx, tpl, p["source"], item)

    def _generated(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str, str | None]:
        return ctx.cache.get(CACHE_KEY, {}).get(f"{tpl['name']}:{src['source_key']}", (None, "", "not prepared", None))

    def _writer(self, ctx: Context, tpl: dict[str, Any], src_key: str, payload: dict[str, Any], status: str,
                chunk_ids: list[str], edge_ids: list[str], results: list[AskResult | Decision], harness_run_id: str | None = None,
                review_reason: str | None = None) -> Writer:
        decision_ids = [d.decision_id for r in results for d in (r.decisions if isinstance(r, AskResult) else [r])]
        example_id = sha256_hex(tpl["name"], src_key, payload)

        async def write(conn) -> str:
            await write_decisions(conn, *results)
            await conn.execute("UPDATE training_examples SET status = 'superseded' WHERE template = %s AND payload->>'source_key' = %s "
                               "AND example_id <> %s AND status <> 'superseded'", (tpl["name"], src_key, example_id))
            await repo.upsert(conn, "training_examples", {
                "example_id": example_id, "template": tpl["name"], "split": None, "payload": {**payload, "source_key": src_key},
                "source_chunk_ids": chunk_ids, "source_edge_ids": edge_ids, "decision_ids": decision_ids,
                "harness_run_id": harness_run_id, "status": status, "registry_version": ctx.registry_version,
                "created_run_id": ctx.run_id}, key=("example_id",))
            if status == IN_REVIEW:
                await enqueue_review(conn, ctx, "training_example", example_id, review_reason or "training_filter_uncertain",
                                     payload={"template": tpl["name"]})
                return REVIEW
            return DONE

        return write

    @staticmethod
    def _combine(outcomes: list[str]) -> str:
        if any(o == REJECT for o in outcomes):
            return REJECTED
        return ACCEPTED if all(o == ACCEPT for o in outcomes) else IN_REVIEW

    async def _missing(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any], reason: str) -> Writer:
        raise RuntimeError(f"harness produced no valid record for {tpl['name']}:{src['source_key']}: {reason}")

    async def _single(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any], item: WorkItem) -> Writer:
        rec, profile, reason, run_id = self._generated(ctx, tpl, src)
        if rec is None:
            return await self._missing(ctx, tpl, src, reason)
        reg = ctx.reg
        chunk = src["text"]
        g = ground(rec["evidence"], chunk, reg.policy("grounding.fuzzy_threshold"), reg.policy("grounding.char_folds"))
        prompt_field, answer_field = tpl["fields"]["prompt"], tpl["fields"]["answer"]
        payload = {"prompt": rec[prompt_field], "answer": rec[answer_field], "evidence": rec["evidence"], "harness_profile": profile}
        if g.kind == UNGROUNDED:
            return self._writer(ctx, tpl, src["source_key"], payload, REJECTED, [src["chunk_id"]], [], [], run_id)
        r = await ctx.jev.ask(tpl["question_sets"]["filter"], {"prompt": rec[prompt_field], "answer": rec[answer_field], "chunk": chunk},
                              "training_example", sha256_hex(tpl["name"], src["source_key"], payload))
        cite = await check_sentences(ctx.jev, [{"text": rec[answer_field], "citation_ids": ["evidence"]}],
                                     {"evidence": chunk[g.start: g.end]}, 1, tpl["name"])
        status = self._combine([r.single.outcome, *(d.outcome for d in cite.decisions)] if cite.decisions else [REJECT])
        return self._writer(ctx, tpl, src["source_key"], payload, status, [src["chunk_id"]], [], [r, *cite.decisions], run_id)

    async def _multihop(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any], item: WorkItem) -> Writer:
        rec, profile, reason, run_id = self._generated(ctx, tpl, src)
        if rec is None:
            return await self._missing(ctx, tpl, src, reason)
        chunk_ids = self._path_chunks(src)
        texts = await self._texts(ctx, chunk_ids)
        payload = {"prompt": rec["question"], "answer": rec["answer"], "hop_evidence": rec["hop_evidence"], "harness_profile": profile}
        subject = sha256_hex(tpl["name"], src["source_key"], payload)
        results: list[AskResult | Decision] = []
        outcomes: list[str] = []
        for i, cid in enumerate(chunk_ids):  # necessity: answerable WITHOUT chunk_i must be false
            rest = [texts[c] for c in chunk_ids if c != cid and c in texts]
            r = await ctx.jev.ask(tpl["question_sets"]["necessity"], {"question": rec["question"], "answer": rec["answer"], "passages": rest},
                                  "training_example", f"{subject}:{i}")
            results.append(r)
            d = r.single
            outcomes.append({ACCEPT: REJECT, REJECT: ACCEPT}.get(d.outcome, d.outcome))  # inverted: "answerable" accepted = bad
        passages = {c: texts.get(c, "") for c in chunk_ids}
        sentences = [{"text": h["span"], "citation_ids": [h["chunk_id"]]} for h in rec["hop_evidence"]]
        cite = await check_sentences(ctx.jev, [{"text": rec["answer"], "citation_ids": chunk_ids}, *sentences], passages,
                                     ctx.reg.policy("run.concurrency.citation_check"), subject)
        results.extend(cite.decisions)
        outcomes.extend(e["outcome"] or REJECT for e in [*cite.kept, *cite.failed])
        return self._writer(ctx, tpl, src["source_key"], payload, self._combine(outcomes), chunk_ids,
                            [e["edge_id"] for e in src["path"]], results, run_id)

    async def _graph_reasoning(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any], item: WorkItem) -> Writer:
        rec, profile, reason, run_id = self._generated(ctx, tpl, src)
        if rec is None:
            return await self._missing(ctx, tpl, src, reason)
        texts = await self._texts(ctx, [src["src_id"], src["dst_id"]])
        rel = relation_definitions(ctx).get(src["rel"], {})
        evidence = {"source": texts.get(src["src_id"], ""), "target": texts.get(src["dst_id"], ""),
                    "relation": f"{src['rel']}: {rel.get('definition', '')}"}
        payload = {"prompt": rec["question"], "answer": rec["answer"], "relation": src["rel"], "harness_profile": profile}
        cite = await check_sentences(ctx.jev, [{"text": rec["answer"], "citation_ids": list(evidence)}], evidence, 1,
                                     sha256_hex(tpl["name"], src["source_key"]))
        status = self._combine([d.outcome for d in cite.decisions] or [REJECT])
        return self._writer(ctx, tpl, src["source_key"], payload, status, [src["src_id"], src["dst_id"]], [src["edge_id"]], cite.decisions, run_id)

    async def _triple(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any], item: WorkItem) -> Writer:
        if src["triple"] == "edge":
            payload = {"subject": src["src_id"], "predicate": src["rel"], "object": src["dst_id"], "directed": src["directed"],
                       "weight": src["weight"], "admitted_by": src["decision_ids"]}
            chunk_ids, edge_ids = [src["src_id"], src["dst_id"]], [src["edge_id"]]
        else:
            payload = {"claim": src["text"], "claim_type": src["claim_type"], "chunk_id": src["chunk_id"], "admitted_by": src["decision_ids"]}
            chunk_ids, edge_ids = [src["chunk_id"]], []
        w = self._writer(ctx, tpl, src["source_key"], payload, ACCEPTED, chunk_ids, edge_ids, [])
        ids = list(src["decision_ids"] or [])

        async def write(conn) -> str:  # already verified: provenance is the upstream decisions
            status = await w(conn)
            await conn.execute("UPDATE training_examples SET decision_ids = %s WHERE example_id = %s",
                               (ids, sha256_hex(tpl["name"], src["source_key"], payload)))
            return status

        return write

    async def _neighbours(self, ctx: Context, query: str, exclude: list[str], k: int) -> list[dict[str, Any]]:
        vec = np.asarray((await ctx.embedder.embed([query]))[0], dtype=np.float32)
        knn_policy = ctx.reg.policy("store.candidate_knn")
        order = knn.order_by(knn_policy, "ORDER BY embedding <=> %(v)s::vector")
        return await knn.fetch(ctx.db, knn_policy,
            "SELECT chunk_id, text FROM (SELECT chunk_id, text, embedding <=> %(v)s::vector AS dist FROM chunks "
            "WHERE corpus_id = %(c)s AND status = 'accepted' AND embedding IS NOT NULL AND NOT (chunk_id = ANY(%(x)s)) "
            f"{order} LIMIT %(fetch)s) s ORDER BY dist, chunk_id LIMIT %(k)s",
            {"c": ctx.corpus_id, "x": exclude, "v": vec, "k": k, "fetch": k * ctx.reg.policy("store.ann_overfetch")})

    async def _retrieval_pair(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any], item: WorkItem) -> Writer:
        query = src["payload"]["prompt"]
        positives = list(src["source_chunk_ids"])
        cands = await self._neighbours(ctx, query, positives, ctx.reg.policy("training.retrieval.negative_candidates"))
        results: list[AskResult] = []
        negatives = []
        for c in cands:
            r = await ctx.jev.ask(tpl["question_sets"]["negative"], {"query": query, "passage": c["text"]}, "negative", sha256_hex(query, c["chunk_id"]))
            results.append(r)
            if r.single.outcome == REJECT:  # Jev: the passage does NOT answer the query → a hard negative
                negatives.append(c["chunk_id"])
        negatives = negatives[: ctx.reg.policy("training.retrieval.max_negatives")]
        payload = {"query": query, "positive_chunk_ids": positives, "negative_chunk_ids": negatives, "from_example": src["example_id"]}
        status = ACCEPTED if negatives else REJECTED
        return self._writer(ctx, tpl, src["source_key"], payload, status, [*positives, *negatives], [], results)

    async def _rerank(self, ctx: Context, tpl: dict[str, Any], src: dict[str, Any], item: WorkItem) -> Writer:
        query = src["payload"]["prompt"]
        cands = await self._neighbours(ctx, query, [], ctx.reg.policy("training.rerank.candidates"))
        items = {c["chunk_id"]: {"passage": c["text"]} for c in cands}
        r = await ctx.jev.ask(tpl["question_sets"]["relevance"], {"query": query}, "rerank_query", src["example_id"], fanout_items=items) \
            if ctx.reg.question_set(tpl["question_sets"]["relevance"]).fanout else None
        results: list[AskResult] = []
        labels = {}
        q_rel = ctx.reg.policy("search.relevance_question")
        if r is not None:
            results.append(r)
            labels = {d.item: d.answers[q_rel]["p"] for d in r.decisions}
        else:
            for cid, it in items.items():
                one = await ctx.jev.ask(tpl["question_sets"]["relevance"], {"query": query, **it}, "rerank_pair", sha256_hex(query, cid),
                                        only={q_rel})
                results.append(one)
                labels[cid] = one.single.answers[q_rel]["p"]
        payload = {"query": query, "candidates": [{"chunk_id": c, "soft_label": p} for c, p in labels.items()], "from_example": src["example_id"]}
        return self._writer(ctx, tpl, src["source_key"], payload, ACCEPTED if labels else REJECTED, list(labels), [], results)

    # ------------------------------------------------------------- finalize

    async def finalize(self, ctx: Context, report: RunReport, opts: RunOptions) -> None:
        await self.link_upstream(ctx)
        await self.dedup(ctx)
        await self.assign_splits(ctx)

    async def link_upstream(self, ctx: Context) -> None:
        """Derived examples inherit the decisions that admitted their source example (R-090)."""
        async with ctx.db.tx() as conn:
            await conn.execute(
                "UPDATE training_examples t SET decision_ids = ARRAY(SELECT DISTINCT d FROM unnest(t.decision_ids || s.decision_ids) d ORDER BY d) "
                "FROM training_examples s WHERE s.example_id = t.payload->>'from_example' AND NOT (t.decision_ids @> s.decision_ids)")

    async def dedup(self, ctx: Context) -> None:
        pol = ctx.reg.policy("training.dedup")
        rows = await ctx.db.fetch(
            "SELECT example_id, template, payload->>'prompt' AS prompt FROM training_examples WHERE status = %s "
            "AND payload ? 'prompt' ORDER BY example_id", (ACCEPTED,))
        by_tpl: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_tpl[r["template"]].append(r)
        for name, group in sorted(by_tpl.items()):
            item = self.item(ctx, f"dedup:{name}", [(r["example_id"], r["prompt"]) for r in group])

            async def factory(group=group) -> Writer:
                vecs = np.asarray(await ctx.embedder.embed([r["prompt"] for r in group]), dtype=float)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.where(norms == 0, 1, norms)
                sims = vecs @ vecs.T
                dropped: set[str] = set()
                results: list[AskResult] = []
                for i in range(len(group)):
                    if group[i]["example_id"] in dropped:
                        continue
                    for j in np.argsort(-sims[i])[: pol["ann_k"] + 1]:
                        j = int(j)
                        if j <= i or sims[i, j] < pol["min_similarity"] or group[j]["example_id"] in dropped:
                            continue
                        r = await ctx.jev.ask(QS_DEDUP, {"question_a": group[i]["prompt"], "question_b": group[j]["prompt"]},
                                              "example_pair", sha256_hex(group[i]["example_id"], group[j]["example_id"]))
                        results.append(r)
                        if r.single.outcome == ACCEPT:
                            dropped.add(group[j]["example_id"])

                async def write(conn) -> str:
                    await write_decisions(conn, *results)
                    if dropped:
                        await conn.execute("UPDATE training_examples SET status = %s WHERE example_id = ANY(%s)", (DUPLICATE, sorted(dropped)))
                    return DONE

                return write

            await run_single_item(ctx, self, item, factory)

    async def assign_splits(self, ctx: Context) -> None:
        pol = ctx.reg.policy("training.splits")
        chunk_doc = {r["chunk_id"]: r["doc_id"] for r in await ctx.db.fetch(
            "SELECT chunk_id, doc_id FROM chunks WHERE corpus_id = %s", (ctx.corpus_id,))}
        edges = [] if pol["by"] == "document" else [(e["src_id"], e["dst_id"]) for e in await self._edges(ctx)]
        keys = component_keys(edges, chunk_doc)
        rows = await ctx.db.fetch("SELECT example_id, source_chunk_ids FROM training_examples WHERE status = %s", (ACCEPTED,))
        async with ctx.db.tx() as conn:
            for r in rows:
                groups = sorted({keys.get(c, c) for c in r["source_chunk_ids"] or []})
                split = assign_split(groups[0] if groups else r["example_id"], pol["fractions"])
                await conn.execute("UPDATE training_examples SET split = %s WHERE example_id = %s", (split, r["example_id"]))


class DerivedTrainingStage(TrainingStage):
    name = "training_derived"
    kinds = DERIVED

    async def finalize(self, ctx: Context, report: RunReport, opts: RunOptions) -> None:
        await self.link_upstream(ctx)
        await self.assign_splits(ctx)


# ----------------------------------------------------------------- export


async def export(ctx: Context, out_dir: Path, fmt: str, templates: list[str] | None = None) -> dict[str, Any]:
    """JSONL / Parquet / HF `datasets`; each record carries full provenance (R-090)."""
    import json

    rows = await ctx.db.fetch(
        "SELECT t.example_id, t.template, t.split, t.payload, t.source_chunk_ids, t.source_edge_ids, t.decision_ids, "
        "t.registry_version, h.harness, h.model AS harness_model FROM training_examples t "
        "LEFT JOIN harness_runs h ON h.harness_run_id = t.harness_run_id WHERE t.status = %s "
        "AND (%s::text[] IS NULL OR t.template = ANY(%s)) ORDER BY t.template, t.split, t.example_id",
        (ACCEPTED, templates, templates))
    records = [{**r, "jev_model": ctx.jev.provider.model} for r in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[(r["template"], r["split"] or "unassigned")].append(r)
    for (tpl, split), recs in sorted(by.items()):
        if fmt == "jsonl":
            path = out_dir / tpl / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) + "\n" for r in recs), encoding="utf-8")
        elif fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            path = out_dir / tpl / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            flat = [{**r, "payload": json.dumps(r["payload"], ensure_ascii=False, sort_keys=True)} for r in recs]
            pq.write_table(pa.Table.from_pylist(flat), path)
        elif fmt == "hf":
            continue
        else:
            raise ValueError(f"unknown export format `{fmt}`")
        written[f"{tpl}/{split}"] = {"path": str(path), "count": len(recs)}
    if fmt == "hf":
        from datasets import Dataset, DatasetDict

        for tpl in sorted({t for t, _ in by}):
            dd = DatasetDict({s: Dataset.from_list([{**r, "payload": json.dumps(r["payload"], ensure_ascii=False, sort_keys=True)} for r in recs])
                              for (t, s), recs in by.items() if t == tpl})
            dd.save_to_disk(str(out_dir / tpl))
            written[tpl] = {"path": str(out_dir / tpl), "splits": {s: len(v) for s, v in dd.items()}}
    return written
