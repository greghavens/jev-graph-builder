"""S2: segmentation by code, then Jev's chunk check (§8.3).

Per document: a heading that follows content starts a new section. A section
that fits the chunk cap (derived from the profiles) is one chunk; a longer one is
split by code at unit boundaries (`segment_spans`). Each chunk's context prefix is
its heading path; its role is code, from its structure (`structural_role`). Jev
is asked, per chunk and packed per document in `QS.chunk_check_fanout`, whether
it carries an injection (the only injection check of the build), whether it is
boilerplate and what its topic is. Structural edges (NEXT, PART_OF,
IN_SECTION) are code.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any

from jev_graph_builder.ids import sha256_hex, short_key
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.jev.service import AskResult, Decision
from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.parse.parsers import CODE, HEADING, LIST, TABLE
from jev_graph_builder.pipeline.common import Context, Deps, Stage, Writer, chunk_metadata, write_decisions
from jev_graph_builder.pipeline.s1_ingest import DOC_JOIN
from jev_graph_builder.pipeline.segment_spans import spans
from jev_graph_builder.store import repo

QS_CHECK_FANOUT = "chunk_check_fanout"
QS_APPLIES, QS_DERIVABLE = "metadata_applies", "metadata_derivable"
_TOKEN = re.compile(r"\w+(?:\.\w+)*")  # a word, or a dotted run such as a version
_ORDERED = re.compile(r"^\s*\d+[.)]\s")  # an ordered list's first item


def max_chunk_tokens(ctx: Context) -> int:
    """§8.3: chunk cap = min(embedding max tokens, Jev state share, chunk check window) × policy fraction.
    Within the check window, Jev's injection check sees every chunk whole."""
    reg = ctx.reg
    emb = reg.profile("embedding", ctx.settings.profiles.embedding)
    jev_share = ctx.jev.profile["context_tokens"] * reg.policy("segment.jev_state_share")
    window = reg.policy("segment.chunk_state_tokens")
    return max(int(min(emb["max_tokens"], jev_share, window) * reg.policy("segment.max_fraction")), 1)


def heading_path_of(unit: dict[str, Any]) -> list[str]:
    path = list(unit["heading_path"] or [])
    return [*path, unit["text"]] if unit["kind"] == HEADING else path


def structural_role(units: list[dict[str, Any]], min_share: float) -> str | None:
    """A chunk's role from its structure: `heading` when it holds only headings; else the kind (ordered
    list -> procedure, table -> reference, code -> example) holding more than `min_share` of its
    non-heading tokens; else none."""
    body = [u for u in units if u["kind"] != HEADING]
    if not body:
        return "heading"
    shares: Counter[str] = Counter()
    for u in body:
        if u["kind"] == LIST and _ORDERED.match(u["text"]):
            shares["procedure"] += u["tokens"]
        elif u["kind"] in (TABLE, CODE):
            shares["reference" if u["kind"] == TABLE else "example"] += u["tokens"]
    if not shares:
        return None
    role, n = shares.most_common(1)[0]
    return role if n > min_share * max(sum(u["tokens"] for u in body), 1) else None


def _tokens(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def named_values(text: str, values: list[str]) -> list[str]:
    """The values `text` names word for word (case-insensitive): a value's tokens appear in order as
    whole tokens, where a dotted token is one ("9.0" is named by "VCF 9.0." but not by "9.0.1" or "19.0")."""
    t = f" {_tokens(text)} "
    return [v for v in values if (k := _tokens(v)) and f" {k} " in t]


class SegmentStage(Stage):
    name = "segment"
    deps = Deps(
        question_sets=(QS_CHECK_FANOUT, QS_APPLIES, QS_DERIVABLE),
        policies=("segment", "jev.untrusted_field", "ingest.metadata_fields", "ingest.applies_values_per_call",
                  "ingest.derivable_values_shown"),
        corpus=True,
        ontology=("topics",),
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

    # -------------------------------------------------------------- chunks

    async def _check_chunks(self, ctx: Context, doc_id: str, drafts: list[dict[str, Any]],
                            results: list[AskResult]) -> list[Decision]:
        """Jev's check of every chunk (injection, boilerplate, topic), a document's chunks packed per call.
        A chunk longer than the check window would reach Jev truncated, its tail never checked for
        injection: the document fails instead."""
        cap = ctx.reg.policy("segment.chunks_per_call")
        window = ctx.reg.policy("segment.chunk_state_tokens")
        truncate = ctx.jev.provider.estimator.truncate
        unseen = [d["chunk_id"] for d in drafts if truncate(d["text"], window, "head") != d["text"]]
        if unseen:
            raise ValueError(f"{doc_id}: {len(unseen)} chunk(s) exceed the {window}-token chunk check window")
        out: list[Decision] = []
        for i in range(0, len(drafts), cap):
            items = {short_key(d["chunk_id"]): {"text": d["text"], "heading_path": heading_path_of(d["units"][0])}
                     for d in drafts[i:i + cap]}
            r = await ctx.jev.ask(QS_CHECK_FANOUT, {}, "chunk_batch", sha256_hex(doc_id, *items), fanout_items=items)
            results.append(r)
            by_item = r.by_item()
            out.extend(by_item[key] for key in items)
        return out

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        reg = ctx.reg
        doc_id = item.item_id
        units = await ctx.db.fetch("SELECT * FROM units WHERE doc_id = %s ORDER BY ord", (doc_id,))
        doc = await ctx.db.fetchone("SELECT corpus_id, title, meta FROM documents WHERE doc_id = %s", (doc_id,))
        meta, results = await self._derive_metadata(ctx, doc_id, doc["title"], units, chunk_metadata(reg, doc["meta"]))

        sep = reg.policy("segment.heading_separator")
        drafts: list[dict[str, Any]] = []
        prev_heading: list[str] = []
        for s, e in spans(units, max_chunk_tokens(ctx)):
            cu = units[s:e]
            text = DOC_JOIN.join(u["text"] for u in cu)
            drafts.append({"units": cu, "text": text, "prefix": sep.join(prev_heading) or None,
                           "chunk_id": sha256_hex(doc_id, cu[0]["unit_id"], cu[-1]["unit_id"], text)})
            prev_heading = heading_path_of(cu[-1])

        checks = await self._check_chunks(ctx, doc_id, drafts, results)

        min_share = reg.policy("segment.role_min_share")
        chunk_rows: list[dict[str, Any]] = []
        for ord_, (d, check) in enumerate(zip(drafts, checks, strict=True)):
            cu = d["units"]
            accepted = check.outcome == ACCEPT
            chunk_rows.append({
                "chunk_id": d["chunk_id"], "corpus_id": doc["corpus_id"], "doc_id": doc_id, "ord": ord_,
                "unit_ids": [u["unit_id"] for u in cu], "heading_path": heading_path_of(cu[0]), "text": d["text"],
                "context_prefix": d["prefix"], "tokens": sum(u["tokens"] for u in cu),
                "role": structural_role(cu, min_share), "topic": check.answers["topic"]["choice"] if accepted else None,
                "boilerplate": ctx.flag_decision(check, "boilerplate"), "meta": meta,
                "status": "accepted" if accepted else "rejected",
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
                                   update=("ord", "context_prefix", "role", "topic", "boilerplate", "meta", "status",
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
