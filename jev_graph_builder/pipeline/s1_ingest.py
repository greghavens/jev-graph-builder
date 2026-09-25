"""S1: ingest and normalize (code) + triage (Jev) (§8.2).

Per file: parse to structural units, normalize, split over-long units (prose
at sentence boundaries, then lines and spaces), compute `doc_id = sha256(normalized_text + source_uri)`, link exact duplicates
with the structural `DUPLICATE_OF` edge, and run `QS.doc_triage`. Out-of-scope
documents are parked. Injection is checked per chunk in S2.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import py3langid
import pysbd

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.parse.parsers import CODE, HEADING, TABLE, ParsedUnit, parse
from jev_graph_builder.pipeline.common import Context, Deps, Stage, Writer, write_decisions
from jev_graph_builder.store import repo

DOC_JOIN = "\n\n"
QS_TRIAGE = "doc_triage"


def discover_origins(paths: list[str]) -> dict[Path, tuple[int, str]]:
    """Every document file under `paths`, resolved, mapped to where it was found: (index of the input path,
    first directory below it, or "" for a file directly in it). The origin is taken before symlinks are
    resolved, so a linked file belongs to the input that links it."""
    out: dict[Path, tuple[int, str]] = {}
    for i, spec in enumerate(paths):
        for m in glob.glob(spec, recursive=True) or [spec]:
            p = Path(m)
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and not f.name.startswith("."):
                        rel = f.relative_to(p).parts
                        out.setdefault(f.resolve(), (i, rel[0] if len(rel) > 1 else ""))
            elif p.is_file():
                out.setdefault(p.resolve(), (i, ""))
    return out


def discover(paths: list[str]) -> list[Path]:
    return sorted(discover_origins(paths))


def detect_language(reg: Any, text: str) -> str:
    """The document's language as an ontology `languages` name: deterministic language ID by code;
    a language the ontology does not list is `ingest.language_fallback`."""
    code = py3langid.classify(text)[0]
    names = {i["name"] for i in reg.ontology("languages")}
    return code if code in names else reg.policy("ingest.language_fallback")


def fit_text(text: str, max_tokens: int, estimate: Any, seps: tuple[str, ...] = ("\n", " ")) -> list[str]:
    """`text` as pieces of at most `max_tokens`: grouped at line breaks, then at spaces, then cut."""
    if estimate(text) <= max_tokens:
        return [text]
    if not seps:
        out, rest = [], text
        while estimate(rest) > max_tokens:
            n = max(len(rest) * max_tokens // estimate(rest), 1)
            out.append(rest[:n])
            rest = rest[n:]
        return [*out, rest] if rest else out
    sep, finer = seps[0], seps[1:]
    out, group = [], ""
    for part in text.split(sep):
        joined = f"{group}{sep}{part}" if group else part
        if estimate(joined) <= max_tokens:
            group = joined
            continue
        if group:
            out.append(group)
        if estimate(part) <= max_tokens:
            group = part
        else:
            out.extend(fit_text(part, max_tokens, estimate, finer))
            group = ""
    if group:
        out.append(group)
    return out


def split_long_units(units: list[ParsedUnit], max_tokens: int, estimate: Any, language: str) -> list[ParsedUnit]:
    """§8.3 step 1: every non-heading unit fits `max_tokens`. Prose becomes sentence groups (a sentence
    still over the cap, and code or a table, is split at lines, then spaces)."""
    seg = pysbd.Segmenter(language=language, clean=False)
    out: list[ParsedUnit] = []
    for u in units:
        if estimate(u.text) <= max_tokens or u.kind == HEADING:
            out.append(u)
            continue
        if u.kind in (CODE, TABLE):
            pieces = fit_text(u.text, max_tokens, estimate)
        else:
            pieces, group = [], []
            for sentence in seg.segment(u.text):
                s = sentence.strip()
                if group and estimate(" ".join([*group, s])) > max_tokens:
                    pieces.append(" ".join(group))
                    group = []
                group.append(s)
            if group:
                pieces.append(" ".join(group))
            pieces = [p for piece in pieces for p in fit_text(piece, max_tokens, estimate)]
        out.extend(ParsedUnit(u.kind, p, u.heading_path, u.page) for p in pieces)
    return out


class IngestStage(Stage):
    name = "ingest"
    deps = Deps(
        question_sets=(QS_TRIAGE,),
        policies=("ingest", "segment.max_unit_tokens"),
        ontology=("doc_types", "languages"),
        corpus=True,
        structural=True,
    )

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        paths = ctx.cache.get("ingest_paths") or ctx.reg.corpus["doc_sources"]
        items = []
        for path in discover(paths):
            uri = path.as_uri()
            items.append(self.item(ctx, uri, sha256_hex(path.read_bytes().hex()), payload=path))
        return items

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        reg = ctx.reg
        path: Path = item.payload
        uri = item.item_id
        parsed = parse(path, reg.policy("ingest"))
        est = ctx.jev.provider.estimator.text
        units = split_long_units(parsed.units, reg.policy("segment.max_unit_tokens"), est, reg.policy("ingest.sentence_language"))
        text = DOC_JOIN.join(u.text for u in units)
        content_hash = sha256_hex(text)
        doc_id = sha256_hex(text, uri)

        unit_rows: list[dict[str, Any]] = []
        offset = 0
        for ord_, u in enumerate(units):
            start = offset
            end = start + len(u.text)
            offset = end + len(DOC_JOIN)
            unit_rows.append({
                "unit_id": sha256_hex(doc_id, ord_, u.text), "doc_id": doc_id, "ord": ord_, "kind": u.kind,
                "heading_path": u.heading_path, "text": u.text, "tokens": est(u.text), "page": u.page,
                "char_start": start, "char_end": end,
            })

        dup = await ctx.db.fetchone(
            "SELECT doc_id FROM documents WHERE corpus_id = %s AND content_hash = %s AND doc_id <> %s "
            "AND status <> 'superseded' ORDER BY doc_id LIMIT 1",
            (ctx.corpus_id, content_hash, doc_id),
        )
        triage = None
        in_scope: bool | None = None
        status = "accepted"
        doc_type = None
        language = detect_language(reg, text) if text else None
        if dup is None and text:
            headings = [u.text for u in units if u.kind == HEADING][: reg.policy("ingest.triage_headings")]
            inputs = {
                "document": {"title": parsed.title or "", "headings": headings, "opening": text},
                "corpus": {"scope": reg.corpus["scope"]},
            }
            triage = await ctx.jev.ask(QS_TRIAGE, inputs, "document", doc_id)
            d = triage.single
            doc_type = d.answers["doc_type"]["choice"]
            if d.outcome == ACCEPT:
                in_scope = True
            else:
                in_scope, status = False, "parked"
        elif dup is not None:
            status = "duplicate"
        doc_row = {
            "doc_id": doc_id, "corpus_id": ctx.corpus_id, "source_uri": uri, "content_hash": content_hash,
            "mime": parsed.mime, "title": parsed.title, "language": language, "doc_type": doc_type,
            "in_scope": in_scope, "meta": parsed.meta,
            "triage_decision_id": triage.single.decision_id if triage else None, "status": status,
            "registry_version": ctx.registry_version, "created_run_id": ctx.run_id,
        }

        async def write(conn) -> str:
            # A changed file gets a new doc_id; the old version and its chunks are superseded, never deleted.
            old = await repo.fetch(
                conn, "UPDATE documents SET status = 'superseded' WHERE corpus_id = %s AND source_uri = %s AND doc_id <> %s "
                "RETURNING doc_id", (ctx.corpus_id, uri, doc_id),
            )
            if old:
                chunks = await repo.fetch(conn, "SELECT chunk_id FROM chunks WHERE doc_id = ANY(%s) AND status <> 'superseded'",
                                          ([r["doc_id"] for r in old],))
                await repo.supersede_chunks(conn, [r["chunk_id"] for r in chunks])
            await repo.upsert(conn, "documents", doc_row, key=("doc_id",))
            await repo.upsert_many(conn, "units", unit_rows, key=("unit_id",))
            await write_decisions(conn, triage)
            if dup is not None:
                rel = reg.structural_label("duplicate_of")
                await repo.upsert(conn, "edges", {
                    "edge_id": sha256_hex(doc_id, dup["doc_id"], rel), "corpus_id": ctx.corpus_id,
                    "src_kind": "document", "src_id": doc_id, "dst_kind": "document", "dst_id": dup["doc_id"],
                    "rel": rel, "directed": True, "structural": True, "weight": None, "features": {},
                    "status": "accepted", "decision_ids": [], "registry_version": ctx.registry_version,
                    "created_run_id": ctx.run_id,
                }, key=("edge_id",))
            return DONE

        return write
