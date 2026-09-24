"""S3: enrichment — the SDE cascade, harness proposes and Jev disposes (§8.4).

1. `prepare` packs chunks into harness batch jobs (`prompts/extract`), the
   harness profile per batch chosen by `QS.harness_route` unless policy pins it.
   The jobs run in the background; each chunk is verified and written as soon as
   its own job finishes, so a stopped run keeps every chunk already written.
2. Every entity span and claim evidence span is grounded in the chunk text
   (`grounding`); ambiguous spans are resolved by Jev (`QS.span_locate`).
3. Every entity and claim is verified by `QS.extract_verify`; summary
   sentences by `QS.summary_verify` (citation check).
4. Jev is the last model to decide: if Jev says no, the item is dropped. No harness re-extracts or
   second-guesses a Jev answer. A chunk whose harness output never validates
   gets no extraction.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pysbd

from jev_graph_builder.ids import sha256_hex, short_key
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.jev.service import AskResult, Decision
from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.pipeline.common import Context, Deps, Stage, Writer, as_text, pending_in, write_decisions
from jev_graph_builder.pipeline.grounding import LOCATED, UNGROUNDED, Grounding, ground, line_of
from jev_graph_builder.store import repo

JOB = "extract"
PROMPT = "extract"
QS_VERIFY, QS_VERIFY_FANOUT, QS_SUMMARY, QS_LOCATE = "extract_verify", "extract_verify_fanout", "summary_verify", "span_locate"
QS_SUMMARY_FANOUT = "summary_verify_fanout"
ENTITY, CLAIM = "entity", "claim"
CACHE_KEY = "s3_records"
READY_KEY, TASKS_KEY = "s3_ready", "s3_tasks"


@dataclass
class Descriptors:
    """Jev-verified chunk descriptors (the parts of the harness record that were accepted)."""

    summary: str | None
    title: str | None
    keywords: list[str]
    decision_ids: list[str]


@dataclass
class Verified:
    kind: str
    data: dict[str, Any]
    grounding: Grounding
    outcome: str
    decisions: list[Decision] = field(default_factory=list)

    @property
    def span(self) -> str:
        return self.data["span"] if self.kind == ENTITY else self.data["evidence_span"]


def items_of(record: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    if not record:
        return []
    return [(ENTITY, e) for e in record.get("entities", [])] + [(CLAIM, c) for c in record.get("claims", [])]


class EnrichStage(Stage):
    name = "enrich"
    deps = Deps(
        question_sets=(QS_VERIFY, QS_VERIFY_FANOUT, QS_SUMMARY, QS_SUMMARY_FANOUT, QS_LOCATE, "harness_route"),
        prompts=(PROMPT,),
        policies=("extract", "grounding", "harness.pinned", "harness.defaults", "segment.heading_separator"),
        ontology=("entity_types", "claim_types"),
        schemas=("extract_record",),
    )

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        rows = await ctx.db.fetch(
            "SELECT c.chunk_id, c.text, c.heading_path, c.meta, d.language, d.title AS doc_title FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE c.corpus_id = %s AND c.status = 'accepted' AND NOT coalesce(c.boilerplate, false) ORDER BY c.doc_id, c.ord",
            (ctx.corpus_id,),
        )
        return [self.item(ctx, r["chunk_id"], r["text"], r["meta"], payload=r) for r in rows]

    # ----------------------------------------------------------- harness jobs

    def _record(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"id": row["chunk_id"], "text": row["text"], "heading_path": row["heading_path"] or [], "metadata": row.get("meta") or {}}

    def _vars(self, ctx: Context) -> dict[str, Any]:
        return {"ontology_file": ctx.reg.policy("extract.ontology_file")}

    def _files(self, ctx: Context) -> dict[str, Any]:
        return {ctx.reg.policy("extract.ontology_file"): {
            "entity_types": ctx.reg.ontology("entity_types"), "claim_types": ctx.reg.ontology("claim_types")}}

    async def prepare(self, ctx: Context, items: list[WorkItem], enumerated: list[WorkItem]) -> None:
        reg = ctx.reg
        size = reg.policy("extract.batch_chunks")
        batches = [enumerated[i: i + size] for i in range(0, len(enumerated), size)]
        sem = asyncio.Semaphore(reg.policy("run.concurrency.harness_jobs"))
        cache: dict[str, tuple[dict[str, Any] | None, str, str]] = ctx.cache.setdefault(CACHE_KEY, {})
        ready: dict[str, asyncio.Future[None]] = ctx.cache.setdefault(READY_KEY, {})
        loop = asyncio.get_running_loop()

        async def run(batch: list[WorkItem], group: list[WorkItem]) -> None:
            try:
                async with sem:
                    records = [self._record(i.payload) for i in group]
                    profile = await ctx.jobs.choose_profile(JOB, {"records": [i.payload["text"] for i in batch]})
                    cap = reg.profile("harness", profile).get("max_batch_records") or len(records)
                    for start in range(0, len(records), cap):
                        part = records[start: start + cap]
                        outcome = await ctx.jobs.run_job(JOB, PROMPT, part, profile, variables=self._vars(ctx), files=self._files(ctx))
                        for r in part:
                            cache[r["id"]] = (outcome.records.get(r["id"]), profile, outcome.missing.get(r["id"], ""))
                            ready[r["id"]].set_result(None)
            except BaseException as exc:
                # The chunks still waiting on this job fail with its error (or stop with it).
                for i in group:
                    f = ready[i.item_id]
                    if not f.done():
                        f.cancel() if isinstance(exc, asyncio.CancelledError) else f.set_exception(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise

        tasks: list[asyncio.Task[None]] = ctx.cache.setdefault(TASKS_KEY, [])
        for b, todo in pending_in(batches, items):
            ready.update({i.item_id: loop.create_future() for i in todo})
            tasks.append(asyncio.ensure_future(run(b, todo)))

    async def cleanup(self, ctx: Context) -> None:
        """Stop the jobs no chunk will wait for any more (a stopped run) and settle their results."""
        tasks: list[asyncio.Task[None]] = ctx.cache.pop(TASKS_KEY, [])
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for f in ctx.cache.pop(READY_KEY, {}).values():
            if f.done() and not f.cancelled():
                f.exception()

    def _state(self, ctx: Context, row: dict[str, Any]) -> dict[str, Any]:
        """What every verification question sees: the whole chunk, what the extractor saw with it, and its task."""
        context = {k: v for k, v in {
            "document_title": row.get("doc_title"),
            "heading_path": ctx.reg.policy("segment.heading_separator").join(row.get("heading_path") or []),
            "metadata": as_text(row.get("meta") or {}),
        }.items() if v}
        return {"chunk": row["text"], "context": context or None, "task": ctx.reg.prompt(PROMPT).task}

    async def _extract_one(self, ctx: Context, row: dict[str, Any]) -> dict[str, Any] | None:
        """A chunk `prepare` did not cover (e.g. a resumed run) gets its own one-record job."""
        profile = await ctx.jobs.choose_profile(JOB, {"records": [row["text"]]})
        out = await ctx.jobs.run_job(JOB, PROMPT, [self._record(row)], profile, variables=self._vars(ctx), files=self._files(ctx))
        return out.records.get(row["chunk_id"])

    # -------------------------------------------------------------- grounding

    async def _ground(self, ctx: Context, row: dict[str, Any], span: str, results: list[AskResult]) -> Grounding:
        reg = ctx.reg
        chunk_id, text = row["chunk_id"], row["text"]
        g = ground(span, text, reg.policy("grounding.fuzzy_threshold"), reg.policy("grounding.char_folds"))
        if not g.ambiguous:
            return g
        lines = {short_key(s, e): (s, e, line_of(text, s, e)) for s, e in g.candidates}
        if len({ln for _, _, ln in lines.values()}) == 1:
            return g  # all candidates sit in the same line: position does not change meaning
        dynamic = {"candidates": {k: ln for k, (_, _, ln) in lines.items()}}
        r = await ctx.jev.ask(QS_LOCATE, {**self._state(ctx, row), "span": span}, "span", sha256_hex(chunk_id, span),
                              dynamic=dynamic)
        results.append(r)
        d = r.single
        pick = d.answers["line"]["choice"]
        if d.outcome == ACCEPT and pick in lines:
            s, e, _ = lines[pick]
            return Grounding(LOCATED, s, e, g.candidates)
        return Grounding(UNGROUNDED, candidates=g.candidates)

    # ----------------------------------------------------------- verification

    def _questions(self, ctx: Context, kind: str, data: dict[str, Any]) -> set[str]:
        only = set(ctx.reg.policy(f"extract.questions.{kind}"))
        if not data.get("attributes"):
            only.discard(ctx.reg.policy("extract.attributes_question"))
        return only

    @staticmethod
    def _item(kind: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"kind": kind, **{k: v for k, v in data.items() if isinstance(v, (str, list, dict))}}

    @staticmethod
    def _types(ctx: Context, kind: str) -> dict[str, dict[str, Any]]:
        ontology = "entity_types" if kind == ENTITY else "claim_types"
        return {"types": {i["name"]: i["definition"] for i in ctx.reg.ontology(ontology)}}

    @staticmethod
    def _judged(kind: str, data: dict[str, Any], g: Grounding, d: Decision) -> Verified:
        if d.outcome == ACCEPT:
            # Jev disposes: its type re-check is the label that enters the graph.
            key = "type" if kind == ENTITY else "claim_type"
            data = {**data, key: d.answers["type_check"]["choice"]}
        return Verified(kind, data, g, d.outcome, [d])

    async def _verify(self, ctx: Context, row: dict[str, Any], kind: str, data: dict[str, Any], g: Grounding,
                      results: list[AskResult]) -> Verified:
        """One item per call (`extract.packing: single`)."""
        r = await ctx.jev.ask(QS_VERIFY, {**self._state(ctx, row), "item": self._item(kind, data)}, kind,
                              sha256_hex(row["chunk_id"], kind, data), dynamic=self._types(ctx, kind),
                              only=self._questions(ctx, kind, data), language=row.get("language"))
        results.append(r)
        return self._judged(kind, data, g, r.single)

    async def _verify_batch(self, ctx: Context, row: dict[str, Any], kind: str,
                            batch: list[tuple[dict[str, Any], Grounding]], results: list[AskResult]) -> list[Verified]:
        """All items of one kind in one call (`extract.packing: fanout`); each item is gated on its own answers."""
        keyed = {f"i{n}": pair for n, pair in enumerate(batch)}
        r = await ctx.jev.ask(QS_VERIFY_FANOUT, self._state(ctx, row), kind,
                              sha256_hex(row["chunk_id"], kind, [d for d, _ in batch]),
                              fanout_items={k: self._item(kind, d) for k, (d, _) in keyed.items()},
                              item_only={k: self._questions(ctx, kind, d) for k, (d, _) in keyed.items()},
                              dynamic=self._types(ctx, kind), language=row.get("language"))
        results.append(r)
        by_item = {d.item: d for d in r.decisions}
        return [self._judged(kind, d, g, by_item[k]) for k, (d, g) in keyed.items()]

    async def _verify_all(self, ctx: Context, row: dict[str, Any], record: dict[str, Any] | None,
                          results: list[AskResult]) -> list[Verified]:
        sem = asyncio.Semaphore(ctx.reg.policy("extract.verify_concurrency"))
        pairs = items_of(record)

        async def ground(kind: str, data: dict[str, Any]) -> Grounding:
            async with sem:
                span = data["span"] if kind == ENTITY else data["evidence_span"]
                return await self._ground(ctx, row, span, results)

        grounds = await asyncio.gather(*(ground(k, d) for k, d in pairs))
        out: list[Verified] = []
        todo: dict[str, list[tuple[dict[str, Any], Grounding]]] = {ENTITY: [], CLAIM: []}
        for (kind, data), g in zip(pairs, grounds, strict=True):
            if g.kind == UNGROUNDED:
                out.append(Verified(kind, data, g, ctx.reg.policy("grounding.ungrounded_outcome")))
            else:
                todo[kind].append((data, g))

        async def single(kind: str, data: dict[str, Any], g: Grounding) -> list[Verified]:
            async with sem:
                return [await self._verify(ctx, row, kind, data, g, results)]

        async def batch(kind: str, part: list[tuple[dict[str, Any], Grounding]]) -> list[Verified]:
            async with sem:
                return await self._verify_batch(ctx, row, kind, part, results)

        if ctx.reg.policy("extract.packing") == "fanout":
            cap = ctx.reg.policy("extract.fanout_per_call")
            jobs = [batch(k, v[i:i + cap]) for k, v in todo.items() for i in range(0, len(v), cap)]
        else:
            jobs = [single(k, d, g) for k, v in todo.items() for d, g in v]
        for part in await asyncio.gather(*jobs):
            out.extend(part)
        return out

    async def _summary(self, ctx: Context, row: dict[str, Any], record: dict[str, Any] | None,
                       results: list[AskResult]) -> Descriptors:
        """P3: the summary (per sentence), title and each keyword are harness output;
        each piece is kept only when `QS.summary_verify` accepts it."""
        record = record or {}
        seg = pysbd.Segmenter(language=ctx.reg.policy("ingest.sentence_language"), clean=False)
        sentences = [s.strip() for s in seg.segment(record.get("summary") or "") if s.strip()]
        title = (record.get("title") or "").strip()
        keywords = [k.strip() for k in record.get("keywords") or [] if k.strip()]
        pieces = list(dict.fromkeys(sentences + ([title] if title else []) + keywords))
        ok: dict[str, Decision] = {}
        if ctx.reg.policy("extract.packing") == "fanout":
            cap = ctx.reg.policy("extract.fanout_per_call")
            parts = [pieces[i:i + cap] for i in range(0, len(pieces), cap)]
            asks = await asyncio.gather(*(
                ctx.jev.ask(QS_SUMMARY_FANOUT, self._state(ctx, row), "summary_sentence", sha256_hex(row["chunk_id"], part),
                            fanout_items={f"s{n}": p for n, p in enumerate(part)})
                for part in parts
            ))
            results.extend(asks)
            for part, r in zip(parts, asks, strict=True):
                by_item = {d.item: d for d in r.decisions}
                ok.update({p: by_item[f"s{n}"] for n, p in enumerate(part) if by_item[f"s{n}"].outcome == ACCEPT})
        else:
            asks = await asyncio.gather(*(
                ctx.jev.ask(QS_SUMMARY, {**self._state(ctx, row), "sentence": s}, "summary_sentence", sha256_hex(row["chunk_id"], s))
                for s in pieces
            ))
            results.extend(asks)
            ok = {s: r.single for s, r in zip(pieces, asks, strict=True) if r.single.outcome == ACCEPT}
        kept = [s for s in sentences if s in ok]
        out_keywords = [k for k in keywords if k in ok]
        return Descriptors(
            summary=" ".join(kept) or None,
            title=title if title in ok else None,
            keywords=out_keywords,
            decision_ids=sorted({ok[s].decision_id for s in kept + out_keywords + ([title] if title in ok else [])}),
        )

    # ---------------------------------------------------------------- process

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        row = item.payload
        chunk_id = row["chunk_id"]
        ready = ctx.cache.get(READY_KEY, {}).get(chunk_id)
        if ready is not None:
            await asyncio.shield(ready)  # this chunk's own extraction job, not the whole stage
        cached = ctx.cache.get(CACHE_KEY, {}).get(chunk_id)
        record = cached[0] if cached else await self._extract_one(ctx, row)
        results: list[AskResult] = []
        verified = await self._verify_all(ctx, row, record, results)
        summary = await self._summary(ctx, row, record, results)
        return self._writer(ctx, row, record, verified, summary, results)

    def _writer(self, ctx: Context, row: dict[str, Any], record: dict[str, Any] | None, verified: list[Verified],
                summary: Descriptors, results: list[AskResult]) -> Writer:
        chunk_id = row["chunk_id"]
        entities, mentions, claims = [], [], []
        base = {"registry_version": ctx.registry_version, "created_run_id": ctx.run_id}
        for v in verified:
            status = "accepted" if v.outcome == ACCEPT else "rejected"
            dec_ids = [d.decision_id for d in v.decisions]
            if v.kind == ENTITY:
                eid = sha256_hex(chunk_id, ENTITY, v.data["name"], v.data["type"])
                entities.append({"entity_id": eid, "corpus_id": ctx.corpus_id, "canonical_name": v.data["name"], "surface": v.data["name"],
                                 "entity_type": v.data["type"], "aliases": [v.data["name"]],
                                 "description": v.data.get("description"), "attributes": repo.Jsonb(v.data.get("attributes") or {}),
                                 "merged_into": None, "status": status, **base})
                mentions.append({"mention_id": sha256_hex(chunk_id, eid, v.grounding.start, v.grounding.end), "chunk_id": chunk_id,
                                 "entity_id": eid, "surface": v.data["name"], "char_start": v.grounding.start,
                                 "char_end": v.grounding.end, "grounded": v.grounding.kind, "status": status, "decision_ids": dec_ids})
            else:
                cid = sha256_hex(chunk_id, CLAIM, v.data["text"])
                claims.append({"claim_id": cid, "chunk_id": chunk_id, "text": v.data["text"], "claim_type": v.data["claim_type"],
                               "status": status, "decision_ids": dec_ids})

        async def write(conn) -> str:
            await write_decisions(conn, *results)
            await repo.upsert_many(conn, "entities", entities, key=("entity_id",), update=("entity_type", "surface", "description", "attributes", "status", "registry_version"))
            await repo.upsert_many(conn, "mentions", mentions, key=("mention_id",))
            await repo.upsert_many(conn, "claims", claims, key=("claim_id",), update=("claim_type", "status", "decision_ids"))
            await conn.execute("UPDATE mentions SET status = 'superseded' WHERE chunk_id = %s AND NOT (mention_id = ANY(%s))",
                               (chunk_id, [m["mention_id"] for m in mentions]))
            await conn.execute("UPDATE claims SET status = 'superseded' WHERE chunk_id = %s AND NOT (claim_id = ANY(%s))",
                               (chunk_id, [c["claim_id"] for c in claims]))
            await conn.execute(
                "UPDATE entities SET status = 'superseded' WHERE entity_id IN (SELECT entity_id FROM mentions WHERE chunk_id = %s "
                "AND status = 'superseded') AND NOT (entity_id = ANY(%s))", (chunk_id, [e["entity_id"] for e in entities]))
            if record is not None:
                await conn.execute(
                    "UPDATE chunks SET summary = %s, title = %s, keywords = %s, summary_decision_ids = %s WHERE chunk_id = %s",
                    (summary.summary, summary.title, summary.keywords, summary.decision_ids, chunk_id))
            return DONE

        return write
