"""S2: segmentation — Jev decides boundaries, code assembles (§8.3).

Per document: a heading that follows content starts a new section, a structural
boundary cut by code. A section that fits the chunk cap (derived from the
profiles) is one chunk. Only inside a longer section does Jev decide each gap,
with `QS.segment` (one gap per call) or `QS.segment_fanout` (a window of units
with one pair of Nouls per gap, R-111): Jev's accept keeps the two sides
together, reject is a chunk boundary. A run Jev kept together that is still
longer than the cap is split by code at the gap Jev was least sure belongs
together, repeatedly, until every chunk fits. Each chunk's context prefix is its
heading path (code); then the injection check (only when S1's triage did not see
the whole document) and classification: packed per document in `QS.chunk_check_fanout` when
`segment.packing` is fanout, else `QS.injection` then `QS.chunk_classify` per
chunk. Structural edges (NEXT, PART_OF, IN_SECTION) are code.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from jev_graph_builder.ids import sha256_hex, short_key
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.jev.service import AskResult, Decision
from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.parse.parsers import HEADING
from jev_graph_builder.pipeline.common import Context, Deps, Stage, Writer, chunk_metadata, write_decisions
from jev_graph_builder.pipeline.s1_ingest import DOC_JOIN, QS_TRIAGE
from jev_graph_builder.pipeline.segment_spans import runs, split_gaps, weakest_gap
from jev_graph_builder.store import repo

QS_SEGMENT, QS_SEGMENT_FANOUT = "segment", "segment_fanout"
QS_INJECTION, QS_CLASSIFY = "injection", "chunk_classify"
QS_CHECK_FANOUT = "chunk_check_fanout"
QS_APPLIES, QS_DERIVABLE = "metadata_applies", "metadata_derivable"
Q_INJECTION = "contains_injection"
_TOKEN = re.compile(r"\w+(?:\.\w+)*")  # a word, or a dotted run such as a version


def max_chunk_tokens(ctx: Context) -> int:
    """§8.3: chunk cap = min(embedding max tokens, Jev state share) × policy fraction."""
    reg = ctx.reg
    emb = reg.profile("embedding", ctx.settings.profiles.embedding)
    jev_share = ctx.jev.profile["context_tokens"] * reg.policy("segment.jev_state_share")
    return max(int(min(emb["max_tokens"], jev_share) * reg.policy("segment.max_fraction")), 1)


def heading_path_of(unit: dict[str, Any]) -> list[str]:
    path = list(unit["heading_path"] or [])
    return [*path, unit["text"]] if unit["kind"] == HEADING else path


def starts_section(units: list[dict[str, Any]], g: int) -> bool:
    """Gap g (between units g and g+1) is a section start: a heading that follows non-heading content."""
    return units[g + 1]["kind"] == HEADING and units[g]["kind"] != HEADING


def section_groups(gaps: list[int], per_call: int) -> list[list[int]]:
    """Jev's gaps in calls of at most `per_call`, never spanning a code-cut section boundary
    (consecutive gap indices belong to one section)."""
    out: list[list[int]] = []
    for g in gaps:
        if out and g == out[-1][-1] + 1 and len(out[-1]) < per_call:
            out[-1].append(g)
        else:
            out.append([g])
    return out


def jev_gaps(units: list[dict[str, Any]], cap: int) -> list[int]:
    """The gaps Jev decides: those inside a section (between code-cut section starts) longer than
    `cap`. A section that fits is one chunk and none of its gaps is asked."""
    n = len(units)
    out: list[int] = []
    for s, e in runs(n, [starts_section(units, g) for g in range(n - 1)]):
        if sum(u["tokens"] for u in units[s:e]) > cap:
            out.extend(range(s, e - 1))
    return out


def keep_strength(decision: Decision) -> float:
    """How sure Jev is that a gap's two sides belong together: its highest yes-probability, since
    either yes keeps them together."""
    return max(a["p"] for a in decision.answers.values() if a["type"] == "noul")


def _tokens(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def named_values(text: str, values: list[str]) -> list[str]:
    """The values `text` names word for word (case-insensitive): a value's tokens appear in order as
    whole tokens, where a dotted token is one ("9.0" is named by "VCF 9.0." but not by "9.0.1" or "19.0")."""
    t = f" {_tokens(text)} "
    return [v for v in values if (k := _tokens(v)) and f" {k} " in t]


def triage_saw_all(ctx: Context, units: list[dict[str, Any]]) -> bool:
    """Whether S1's triage showed Jev the whole document (its injection answer then covers every
    chunk): the document text survives `doc_triage`'s own truncation of `document.opening` unchanged."""
    spec = ctx.reg.question_set(QS_TRIAGE).state_template["document"]
    text = DOC_JOIN.join(u["text"] for u in units)
    rule = spec.get("truncate", ctx.reg.policy("jev.default_truncation"))
    return ctx.jev.provider.estimator.truncate(text, ctx.reg.policy(spec["max_tokens_ref"]), rule) == text


class SegmentStage(Stage):
    name = "segment"
    deps = Deps(
        question_sets=(QS_SEGMENT, QS_SEGMENT_FANOUT, QS_INJECTION, QS_CLASSIFY, QS_CHECK_FANOUT, QS_APPLIES,
                       QS_DERIVABLE, QS_TRIAGE),
        policies=("segment", "jev.untrusted_field", "jev.default_truncation", "ingest.triage_state_tokens", "ingest.metadata_fields", "ingest.applies_values_per_call",
                  "ingest.derivable_values_shown"),
        corpus=True,
        ontology=("chunk_roles", "topics"),
        embedding=True,
        structural=True,
    )

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        rows = await ctx.db.fetch(
            "SELECT d.doc_id, d.content_hash, array_agg(u.unit_id ORDER BY u.ord) AS unit_ids FROM documents d "
            "JOIN units u ON u.doc_id = d.doc_id WHERE d.corpus_id = %s AND d.status = 'accepted' "
            "AND d.in_scope AND NOT d.quarantined GROUP BY d.doc_id, d.content_hash ORDER BY d.doc_id",
            (ctx.corpus_id,),
        )
        return [self.item(ctx, r["doc_id"], r["content_hash"], r["unit_ids"]) for r in rows]

    # ------------------------------------------------------------ metadata

    async def _field_values(self, ctx: Context, field: str) -> list[str]:
        """The distinct values of a metadata key across the corpus's accepted documents (fixed after S1)."""
        cache = ctx.cache.setdefault("metadata_values", {})
        if field not in cache:
            cache[field] = asyncio.ensure_future(self._query_values(ctx, field))
        return await cache[field]

    async def _query_values(self, ctx: Context, field: str) -> list[str]:
        rows = await ctx.db.fetch(
            "SELECT DISTINCT v FROM documents d, LATERAL jsonb_array_elements_text(CASE jsonb_typeof(d.meta -> %s) "
            "WHEN 'array' THEN d.meta -> %s ELSE jsonb_build_array(d.meta -> %s) END) AS v "
            "WHERE d.corpus_id = %s AND d.status = 'accepted' AND d.in_scope AND d.meta ? %s ORDER BY v",
            (field, field, field, ctx.corpus_id, field),
        )
        return [r["v"] for r in rows if r["v"]]

    async def _derivable(self, ctx: Context, field: str, values: list[str]) -> AskResult:
        """Jev decides once per key whether its values are categories a text can name (a release) rather
        than per-document identifiers (a title); only then are documents lacking it given values."""
        cache = ctx.cache.setdefault(QS_DERIVABLE, {})
        if field not in cache:
            shown = ctx.reg.policy("ingest.derivable_values_shown")
            step = max(len(values) / shown, 1)
            sample = [values[int(i * step)] for i in range(min(shown, len(values)))]
            cache[field] = asyncio.ensure_future(ctx.jev.ask(
                QS_DERIVABLE,
                {"corpus": {"scope": ctx.reg.corpus.get("scope", "")},
                 "field": {"name": field, "distinct_values": str(len(values)), "values": sample}},
                "metadata_field", sha256_hex(field, *values),
            ))
        return await cache[field]

    async def _derive_metadata(self, ctx: Context, doc_id: str, title: str | None, units: list[dict[str, Any]],
                               meta: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[AskResult]]:
        """§8.1: a `metadata_fields` key the document does not carry gets the corpus values its text
        names word for word (code). Only when it names none does Jev say which values its text applies
        to (e.g. through a range or a description). Jev's no → not applied."""
        meta = dict(meta or {})
        results: list[AskResult] = []
        text = DOC_JOIN.join(u["text"] for u in units)
        named_in = DOC_JOIN.join(t for t in (title, text) if t)
        for field in ctx.reg.policy("ingest.metadata_fields"):
            if field in meta:
                continue
            values = await self._field_values(ctx, field)
            derivable = await self._derivable(ctx, field, values)
            results.append(derivable)
            if derivable.single.outcome != ACCEPT:
                continue
            applied = named_values(named_in, values) or await self._jev_applies(ctx, doc_id, title, text, field, values, results)
            if applied:
                meta[field] = applied
        return meta or None, results

    async def _jev_applies(self, ctx: Context, doc_id: str, title: str | None, text: str, field: str,
                           values: list[str], results: list[AskResult]) -> list[str]:
        """The values Jev says the document's text applies to, in fan-out groups of candidate values."""
        per_call = ctx.reg.policy("ingest.applies_values_per_call")
        applied: list[str] = []
        for start in range(0, len(values), per_call):
            group = values[start: start + per_call]
            r = await ctx.jev.ask(
                QS_APPLIES,
                {"corpus": {"scope": ctx.reg.corpus.get("scope", "")}, "field": {"name": field},
                 "document": {"title": title or "", "text": text}},
                "document_metadata", sha256_hex(doc_id, field, *group),
                fanout_items={short_key(field, v): {"value": v} for v in group},
            )
            results.append(r)
            chosen = {d.item for d in r.decisions if d.outcome == ACCEPT}
            applied.extend(v for v in group if short_key(field, v) in chosen)
        return applied

    # ---------------------------------------------------------------- gaps

    async def _gap_cuts(self, ctx: Context, doc_id: str, units: list[dict[str, Any]],
                        cap: int) -> tuple[list[bool], list[float], list[AskResult]]:
        """Per gap: whether it is a boundary, and how sure Jev is its sides belong together (1.0
        where code keeps a fitting section whole, 0.0 at a code-cut section start)."""
        reg = ctx.reg
        k = reg.policy("segment.window_units")
        n = len(units)
        results: list[AskResult] = []
        # A heading that follows content opens a new section: a structural boundary, cut by code.
        # A section that fits the cap stays whole; Jev decides the gaps of longer sections only.
        cut_at = {g: starts_section(units, g) for g in range(n - 1)}
        keep = [0.0 if cut else 1.0 for cut in cut_at.values()]
        gaps = jev_gaps(units, cap)
        use_fanout = reg.policy("segment.packing") == "fanout"
        if not use_fanout:
            for g in gaps:
                before, after = units[max(g - k + 1, 0): g + 1], units[g + 1: g + 1 + k]
                inputs = {
                    "before": DOC_JOIN.join(u["text"] for u in before),
                    "after": DOC_JOIN.join(u["text"] for u in after),
                    "heading_path_before": heading_path_of(units[g]),
                    "heading_path_after": heading_path_of(units[g + 1]),
                }
                r = await ctx.jev.ask(QS_SEGMENT, inputs, "gap", sha256_hex(units[g]["unit_id"], units[g + 1]["unit_id"]))
                results.append(r)
                cut_at[g] = r.single.outcome != ACCEPT
                keep[g] = keep_strength(r.single)
            return [cut_at[g] for g in range(n - 1)], keep, results

        per_call = reg.policy("segment.gaps_per_call")
        for group in section_groups(gaps, per_call):
            lo, hi = max(group[0] - k + 1, 0), min(group[-1] + k + 1, n)
            window = units[lo:hi]
            ukey = {u["unit_id"]: short_key(u["unit_id"]) for u in window}
            state_units = {ukey[u["unit_id"]]: {"text": u["text"], "heading_path": heading_path_of(u)} for u in window}
            items = {}
            gap_keys = []
            for g in group:
                gk = short_key(units[g]["unit_id"], units[g + 1]["unit_id"])
                gap_keys.append(gk)
                items[gk] = {
                    "before": ukey[units[g]["unit_id"]], "after": ukey[units[g + 1]["unit_id"]],
                    "heading_changed": heading_path_of(units[g]) != units[g + 1]["heading_path"],
                }
            r = await ctx.jev.ask(QS_SEGMENT_FANOUT, {"units": state_units}, "gap_window",
                                  sha256_hex(doc_id, *gap_keys), fanout_items=items)
            results.append(r)
            by_item = r.by_item()
            for g, gk in zip(group, gap_keys, strict=True):
                cut_at[g] = by_item[gk].outcome != ACCEPT
                keep[g] = keep_strength(by_item[gk])
        return [cut_at[g] for g in range(n - 1)], keep, results

    @staticmethod
    def _fit(tokens: list[int], keep: list[float], start: int, end: int, max_tokens: int) -> list[tuple[int, int]]:
        """Split a run Jev kept together until every piece fits, each time at the fitting gap Jev was
        least sure belongs together."""
        out: list[tuple[int, int]] = []
        while end - start > 1 and sum(tokens[start:end]) > max_tokens:
            cut = weakest_gap(split_gaps(tokens, start, end, max_tokens), keep)
            out.append((start, cut + 1))
            start = cut + 1
        out.append((start, end))
        return out

    # -------------------------------------------------------------- chunks

    async def _check_chunks(self, ctx: Context, doc_id: str, drafts: list[dict[str, Any]], check_injection: bool,
                            results: list[AskResult | None]) -> list[tuple[bool, Decision | None]]:
        """Per chunk: (Jev flags an injection, the classification decision for a clean chunk). The
        chunk-level injection question is asked only when S1's triage did not see the whole document."""
        if ctx.reg.policy("segment.packing") == "fanout":
            cap = ctx.reg.policy("segment.chunks_per_call")
            asked = set(ctx.reg.question_set(QS_CHECK_FANOUT).questions)
            if not check_injection:
                asked.discard(Q_INJECTION)
            out: list[tuple[bool, Decision | None]] = []
            for i in range(0, len(drafts), cap):
                part = drafts[i:i + cap]
                items = {short_key(d["chunk_id"]): {"text": d["text"], "heading_path": heading_path_of(d["units"][0])} for d in part}
                r = await ctx.jev.ask(QS_CHECK_FANOUT, {}, "chunk_batch", sha256_hex(doc_id, *items), fanout_items=items,
                                      item_only={k: asked for k in items})
                results.append(r)
                by_item = r.by_item()
                for key in items:
                    d = by_item[key]
                    injected = Q_INJECTION in d.answers and ctx.flag_decision(d, Q_INJECTION)
                    out.append((injected, None if injected else d))
            return out

        async def one(d: dict[str, Any]) -> tuple[bool, Decision | None]:
            if check_injection:
                inj = await ctx.jev.ask(QS_INJECTION, {"chunk": d["text"]}, "chunk", d["chunk_id"])
                results.append(inj)
                if ctx.flag(inj, Q_INJECTION):
                    return True, None
            cls = await ctx.jev.ask(QS_CLASSIFY, {"chunk": d["text"], "heading_path": heading_path_of(d["units"][0])},
                                    "chunk", d["chunk_id"])
            results.append(cls)
            return False, cls.single

        return list(await asyncio.gather(*(one(d) for d in drafts)))

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        reg = ctx.reg
        doc_id = item.item_id
        units = await ctx.db.fetch("SELECT * FROM units WHERE doc_id = %s ORDER BY ord", (doc_id,))
        doc = await ctx.db.fetchone("SELECT corpus_id, title, meta FROM documents WHERE doc_id = %s", (doc_id,))
        meta, applies_results = await self._derive_metadata(ctx, doc_id, doc["title"], units, chunk_metadata(reg, doc["meta"]))
        cap = max_chunk_tokens(ctx)
        cuts, keep, gap_results = await self._gap_cuts(ctx, doc_id, units, cap)
        results: list[AskResult | None] = [*applies_results, *gap_results]
        tokens = [u["tokens"] for u in units]
        spans = [piece for s, e in runs(len(units), cuts) for piece in self._fit(tokens, keep, s, e, cap)]

        sep = reg.policy("segment.heading_separator")
        drafts: list[dict[str, Any]] = []
        prev_heading: list[str] = []
        for s, e in spans:
            cu = units[s:e]
            text = DOC_JOIN.join(u["text"] for u in cu)
            drafts.append({"units": cu, "text": text, "prefix": sep.join(prev_heading) or None,
                           "chunk_id": sha256_hex(doc_id, cu[0]["unit_id"], cu[-1]["unit_id"], text)})
            prev_heading = heading_path_of(cu[-1])

        checks = await self._check_chunks(ctx, doc_id, drafts, not triage_saw_all(ctx, units), results)

        chunk_rows: list[dict[str, Any]] = []
        for ord_, (d, (injected, cls)) in enumerate(zip(drafts, checks, strict=True)):
            cu = d["units"]
            role = topic = density = boilerplate = None
            if cls is not None:
                density = cls.answers["density"]["score"]
                boilerplate = ctx.flag_decision(cls, "boilerplate")
                if cls.outcome == ACCEPT:
                    role, topic = cls.answers["role"]["choice"], cls.answers["topic"]["choice"]
            chunk_rows.append({
                "chunk_id": d["chunk_id"], "corpus_id": doc["corpus_id"], "doc_id": doc_id, "ord": ord_,
                "unit_ids": [u["unit_id"] for u in cu], "heading_path": heading_path_of(cu[0]), "text": d["text"],
                "context_prefix": d["prefix"], "tokens": sum(u["tokens"] for u in cu), "role": role, "topic": topic,
                "density": density, "boilerplate": boilerplate, "meta": meta, "status": "rejected" if injected else "accepted",
                "registry_version": ctx.registry_version, "created_run_id": ctx.run_id,
            })

        edges = self._structural_edges(ctx, doc_id, chunk_rows)

        async def write(conn) -> str:
            ids = [c["chunk_id"] for c in chunk_rows]
            stale = await repo.fetch(conn, "SELECT chunk_id FROM chunks WHERE doc_id = %s AND NOT (chunk_id = ANY(%s)) AND status <> 'superseded'", (doc_id, ids))
            stale_ids = [r["chunk_id"] for r in stale]
            await repo.supersede(conn, "chunks", "chunk_id", stale_ids)
            if stale_ids:
                await conn.execute("UPDATE edges SET status = 'superseded' WHERE src_id = ANY(%s) OR dst_id = ANY(%s)", (stale_ids, stale_ids))
            # Chunk text is immutable per chunk_id; keep S3/S4 outputs if the row exists.
            await repo.upsert_many(conn, "chunks", chunk_rows, key=("chunk_id",),
                                   update=("ord", "context_prefix", "role", "topic", "density", "boilerplate", "meta", "status",
                                           "registry_version"))
            await repo.upsert_many(conn, "edges", edges, key=("edge_id",))
            await write_decisions(conn, *results)
            return DONE

        return write

    def _structural_edges(self, ctx: Context, doc_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reg = ctx.reg
        base = {"corpus_id": ctx.corpus_id, "structural": True, "weight": None, "features": {}, "status": "accepted",
                "decision_ids": [], "registry_version": ctx.registry_version, "created_run_id": ctx.run_id, "directed": True}
        out = []

        def edge(src_kind: str, src: str, dst_kind: str, dst: str, role: str) -> None:
            rel = reg.structural_label(role)
            out.append({**base, "edge_id": sha256_hex(src, dst, rel), "src_kind": src_kind, "src_id": src,
                        "dst_kind": dst_kind, "dst_id": dst, "rel": rel})

        for a, b in zip(chunks, chunks[1:], strict=False):
            edge("chunk", a["chunk_id"], "chunk", b["chunk_id"], "next")
        for c in chunks:
            edge("chunk", c["chunk_id"], "document", doc_id, "part_of")
            if c["heading_path"]:
                edge("chunk", c["chunk_id"], "section", sha256_hex(doc_id, c["heading_path"]), "in_section")
        return out
