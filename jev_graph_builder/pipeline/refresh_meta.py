"""Metadata refresh: recompute the document and chunk metadata of an existing build with the current code, without
asking Jev and without touching anything else (text, chunk IDs, embeddings, entities, links).

A document's metadata is read again from its file (a parser that now reads more, e.g. an OpenAPI spec's own
version). Each chunk then gets what segmentation would give it now: the `metadata_fields` its document carries,
else the corpus values its document names word for word (code), else the value it already has (an earlier Jev
decision is kept, never re-asked). A key Jev judged not derivable is never derived. A chunk reused by several
documents carries the metadata of each.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from psycopg.types.json import Jsonb

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.parse.parsers import ParseError, parse
from jev_graph_builder.pipeline.common import Context, as_text, chunk_metadata
from jev_graph_builder.pipeline.s1_ingest import DOC_JOIN, split_long_units
from jev_graph_builder.pipeline.s2_segment import named_values, widened, with_parents
from jev_graph_builder.store import repo



def _path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    return Path(unquote(parsed.path)) if parsed.scheme == "file" else None


def _same(a: Any, b: Any) -> bool:
    """The same values, in any order."""
    if isinstance(a, list) and isinstance(b, list):
        return Counter(map(str, a)) == Counter(map(str, b))
    return a == b


def _values(metas: list[dict[str, Any] | None], field: str) -> list[str]:
    out: set[str] = set()
    for m in metas:
        v = (m or {}).get(field)
        out.update(x for x in (v if isinstance(v, list) else [v]) if x)
    return sorted(out)


async def refresh_metadata(ctx: Context, apply: bool) -> dict[str, Any]:
    reg = ctx.reg
    fields = reg.policy("ingest.metadata_fields")
    docs = await ctx.db.fetch(
        "SELECT doc_id, source_uri, title, meta, content_hash FROM documents WHERE corpus_id = %s AND status = 'accepted' "
        "AND in_scope AND NOT quarantined ORDER BY doc_id", (ctx.corpus_id,))

    # 1. Document metadata, read again from each file. A file that changed since it was ingested (its text no
    #    longer the document's) is a new version for ingest to take in, not a source for this document's metadata.
    est = ctx.jev.provider.estimator.text
    doc_meta: dict[str, dict[str, Any] | None] = {}
    unreadable, file_changed = 0, 0
    for d in docs:
        path = _path(d["source_uri"])
        meta = d["meta"]
        if path is not None and path.is_file():
            try:
                parsed = parse(path, reg.policy("ingest"))
                units = split_long_units(parsed.units, reg.policy("segment.max_unit_tokens"), est,
                                         reg.policy("ingest.sentence_language"))
                if sha256_hex(DOC_JOIN.join(u.text for u in units)) == d["content_hash"]:
                    meta = parsed.meta
                else:
                    file_changed += 1
            except ParseError:
                unreadable += 1
        else:
            unreadable += 1
        doc_meta[d["doc_id"]] = meta
    changed_docs = {d["doc_id"]: doc_meta[d["doc_id"]] for d in docs if doc_meta[d["doc_id"]] != d["meta"]}
    old_doc = {d["doc_id"]: d["meta"] for d in docs}

    # 2. A key is derived only where segmentation derived it before (Jev judged it derivable); no Jev call here.
    derived_rows = await ctx.db.fetch(
        "SELECT DISTINCT f AS field FROM chunks c JOIN documents d USING (doc_id), unnest(%s::text[]) AS f "
        "WHERE c.corpus_id = %s AND c.status = 'accepted' AND c.meta ? f AND NOT coalesce(d.meta ? f, false)",
        (fields, ctx.corpus_id))
    derivable = {r["field"] for r in derived_rows}
    corpus_values = {f: _values([chunk_metadata(reg, m) for m in doc_meta.values()], f) for f in fields}

    # 3. Each document's chunk metadata. A key the document carries is its own value (with the broader values it
    #    extends). A derived key only gains: the values already there (an earlier Jev decision is never dropped)
    #    plus those its text names word for word, with the broader values they extend.
    chunks = await ctx.db.fetch(
        "SELECT chunk_id, doc_id, meta FROM chunks WHERE corpus_id = %s AND status = 'accepted'", (ctx.corpus_id,))
    current_of_doc: dict[str, dict[str, Any]] = {}
    for c in chunks:
        current_of_doc.setdefault(c["doc_id"], c["meta"] or {})
    titles = {d["doc_id"]: d["title"] for d in docs}
    meta_of_doc: dict[str, dict[str, Any] | None] = {}
    for doc_id, dm in doc_meta.items():
        meta = dict(chunk_metadata(reg, dm) or {})
        for f in list(meta):
            meta[f] = widened(meta[f], corpus_values[f])
        missing = [f for f in fields if f not in meta and f in derivable]
        if missing:
            units = await ctx.db.fetch("SELECT text FROM units WHERE doc_id = %s ORDER BY ord", (doc_id,))
            named_in = DOC_JOIN.join(t for t in (titles[doc_id], *(u["text"] for u in units)) if t)
            for f in missing:
                had = current_of_doc.get(doc_id, {}).get(f)
                had = [] if had is None else had if isinstance(had, list) else [had]
                found = [*had, *(v for v in named_values(named_in, corpus_values[f]) if v not in had)]
                if found:
                    meta[f] = with_parents(found, corpus_values[f]) if all(isinstance(v, str) for v in found) else found
        meta_of_doc[doc_id] = as_text(meta) or None

    # 4. Every accepted chunk: its document's metadata, merged with each document that reuses it. The same values
    #    in another order are no change.
    occ = await ctx.db.fetch(
        "SELECT o.chunk_id, o.doc_id FROM chunk_occurrences o JOIN documents d USING (doc_id) WHERE d.status = 'accepted'")
    also: dict[str, list[str]] = {}
    for o in occ:
        also.setdefault(o["chunk_id"], []).append(o["doc_id"])
    changed_chunks: dict[str, dict[str, Any] | None] = {}
    moves: dict[str, Counter[tuple[str, str]]] = {}
    for c in chunks:
        if c["doc_id"] not in meta_of_doc:
            continue
        new = repo.merge_meta(meta_of_doc[c["doc_id"]], *(meta_of_doc.get(d) for d in also.get(c["chunk_id"], [])))
        old = c["meta"] or {}
        new = {f: (old[f] if f in old and _same(old[f], v) else v) for f, v in (new or {}).items()} or None
        if new != c["meta"]:
            changed_chunks[c["chunk_id"]] = new
            for f in fields:
                old_v, new_v = old.get(f), (new or {}).get(f)
                if old_v != new_v:
                    moves.setdefault(f, Counter())[(str(old_v), str(new_v))] += 1

    if apply:
        async with ctx.db.tx() as conn:
            for doc_id, m in changed_docs.items():
                await conn.execute("UPDATE documents SET meta = %s WHERE doc_id = %s", (Jsonb(m) if m else None, doc_id))
            for chunk_id, m in changed_chunks.items():
                await conn.execute("UPDATE chunks SET meta = %s WHERE chunk_id = %s", (Jsonb(m) if m else None, chunk_id))

    return {
        "applied": apply,
        "documents": len(docs), "documents_unreadable": unreadable, "documents_file_changed_since_ingest": file_changed,
        "documents_changed": len(changed_docs),
        "document_changes": [{"title": titles[k], **{f: {"from": (old_doc[k] or {}).get(f), "to": (m or {}).get(f)}
                                                   for f in {*(old_doc[k] or {}), *(m or {})}
                                                   if (old_doc[k] or {}).get(f) != (m or {}).get(f)}}
                             for k, m in changed_docs.items()],
        "chunks": len(chunks), "chunks_changed": len(changed_chunks),
        "changes": {f: [{"from": a, "to": b, "chunks": n} for (a, b), n in cnt.most_common()]
                    for f, cnt in moves.items()},
    }
