"""S0: Registry bootstrap (§8.1, R-080).

1. Code draws a stratified sample of the corpus files (file type × size bucket
   × directory), sized by `policies.bootstrap.sample`.
2. A harness job drafts the ontology, extraction notes, initial policy values
   and open decisions from the sample and `corpus.yaml` (`prompts/bootstrap_registry`).
3. Jev checks every definition against its cited sample passages and every
   relation pair for overlap; overlapping pairs go back to the harness for
   disambiguation, up to `policies.bootstrap.max_disambiguation_rounds`.
4. The draft becomes a *proposed* Registry version (never the active one) and a
   `registry_versions` row with the open decisions, overlaps and unsupported
   definitions for the human gate (`registry approve` / `registry activate`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from jev_graph_builder.ids import sha256_hex, unit_fraction
from jev_graph_builder.ledger.ledger import REVIEW
from jev_graph_builder.parse.parsers import ParseError, parse, repeated_elements
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.pipeline.common import Context, Deps, Stage, as_text, ensure_corpus, run_single_item, write_decisions
from jev_graph_builder.pipeline.ontology_checks import CheckReport, check_definitions
from jev_graph_builder.pipeline.s1_ingest import discover_origins
from jev_graph_builder.registry.gate import BOOTSTRAP_KIND
from jev_graph_builder.registry.loader import ONTOLOGY_KINDS, Registry
from jev_graph_builder.registry.versioning import stage_proposal, start_draft
from jev_graph_builder.store import repo

PROMPT_DRAFT, SCHEMA_DRAFT = "bootstrap_registry", "registry_draft"
QS_METADATA, QS_CHROME = "metadata_field", "chrome_selector"
PROMPT_DISAMBIGUATE, SCHEMA_DISAMBIGUATE = "disambiguate_relations", "relation_proposals"
RELATIONS = "relation_types"
SCOPE_KIND = "corpus_scope"


def sample_files(origins: dict[Path, tuple[int, str]], per_stratum: int, max_total: int, size_buckets: list[int]) -> list[Path]:
    """Stratify by input path × directory below it × suffix × size bucket, `per_stratum` files each, in a
    deterministic order. The `max_total` cap is filled round-robin across input paths, then across strata
    within each input, so every input given to the command is represented."""
    strata: dict[tuple[int, str, str, int], list[Path]] = defaultdict(list)
    for f, (root, top) in origins.items():
        bucket = sum(f.stat().st_size > b for b in size_buckets)
        strata[(root, top, f.suffix.lower(), bucket)].append(f)
    by_root: dict[int, list[list[Path]]] = defaultdict(list)
    for key in sorted(strata, key=lambda k: (k[0], unit_fraction("bootstrap-stratum", repr(k)))):
        group = sorted(strata[key], key=lambda p: unit_fraction("bootstrap", str(p)))
        by_root[key[0]].append(group[:per_stratum])
    queues = [[f for rank in range(per_stratum) for g in groups if rank < len(g) for f in [g[rank]]]
              for _, groups in sorted(by_root.items())]
    picked: list[Path] = []
    while len(picked) < max_total and any(queues):
        for q in queues:
            if q and len(picked) < max_total:
                picked.append(q.pop(0))
    return sorted(picked)


class BootstrapStage(Stage):
    name = "bootstrap"
    deps = Deps(prompts=(PROMPT_DRAFT, PROMPT_DISAMBIGUATE), schemas=(SCHEMA_DRAFT, SCHEMA_DISAMBIGUATE),
                question_sets=("ontology_support", "relation_support", "relation_overlap", QS_METADATA, QS_CHROME),
                policies=("bootstrap", "ingest.html"), corpus=True)

    def passages(self, ctx: Context, paths: list[str]) -> dict[str, dict[str, Any]]:
        pol = ctx.reg.policy("bootstrap.sample")
        chosen = sample_files(discover_origins(paths), pol["per_stratum"], pol["max_documents"], pol["size_buckets"])
        opts = ctx.reg.policy("ingest")
        out: dict[str, dict[str, Any]] = {}
        self.html_pages: list[str] = []
        for f in chosen:
            try:
                doc = parse(f, opts)
            except ParseError:
                continue
            if doc.mime == "text/html":
                self.html_pages.append(Path(f).read_text(encoding="utf-8", errors="replace"))
            units = [u.text for u in doc.units][: pol["units_per_document"]]
            for i, text in enumerate(units):
                pid = f"s{len(out) + 1}"
                out[pid] = {"id": pid, "path": str(f), "unit": i, "text": text[: pol["max_passage_chars"]]}
                if i == 0 and doc.meta:
                    out[pid]["metadata"] = as_text(doc.meta)
        return out

    async def run(self, ctx: Context, paths: list[str]) -> dict[str, Any]:
        """Returns the proposal summary; the active Registry is never modified here."""
        await ensure_corpus(ctx)
        passages = self.passages(ctx, paths)
        item = self.item(ctx, f"bootstrap:{ctx.corpus_id}", sorted((p["path"], p["unit"], sha256_hex(p["text"], p.get("metadata"))) for p in passages.values()))
        summary: dict[str, Any] = {}

        async def factory():
            reg = ctx.reg
            profile = await ctx.jobs.choose_profile(PROMPT_DRAFT, {"records": [p["text"] for p in passages.values()]})
            ws = ctx.jobs.workspace_for(self.name, item.input_hash)
            ws.mkdir(parents=True, exist_ok=True)
            result, run_id = await ctx.jobs.run_single(
                PROMPT_DRAFT, SCHEMA_DRAFT, profile,
                {"corpus": reg.corpus, "passages": list(passages.values()), "current": {k: reg.ontology(k) for k in ONTOLOGY_KINDS}}, ws)
            if not result.ok or not isinstance(result.structured, dict):
                raise RuntimeError(f"bootstrap draft failed: {result.error}")
            draft = result.structured
            texts = {pid: p["text"] for pid, p in passages.items()}
            checks, decisions, rounds = await self._verify(ctx, draft, texts, profile, ws)
            fields, field_decisions = await self._check_metadata_fields(ctx, draft, passages)
            decisions.extend(field_decisions)
            chrome, chrome_decisions = await self._check_chrome(ctx, draft)
            decisions.extend(chrome_decisions)
            version = self._stage(ctx, draft, passages)
            summary.update({
                "kind": BOOTSTRAP_KIND, "version": version, "scope": draft["corpus"], "open_decisions": draft.get("open_decisions", []),
                "drafted": {k: [i["name"] for i in draft["ontology"].get(k) or []] for k in ONTOLOGY_KINDS},
                "unsupported": [u for c in checks.values() for u in c.unsupported],
                "overlaps": checks[RELATIONS].overlaps if RELATIONS in checks else [],
                "metadata_fields": fields,
                "strip_selectors": chrome,
                "disambiguation_rounds": rounds, "harness_run_id": run_id, "sample": sorted({p["path"] for p in passages.values()}),
            })

            async def write(conn) -> str:
                await write_decisions(conn, *decisions)
                await repo.upsert(conn, "registry_versions", {"version": version, "corpus_id": ctx.corpus_id, "status": "proposed",
                                                              "summary": summary}, key=("version",), update=("summary",))
                return REVIEW  # the human gate is always required

            return write

        await run_single_item(ctx, self, item, factory)
        return summary

    async def _verify(self, ctx: Context, draft: dict[str, Any], texts: dict[str, str], profile: str, ws: Path
                      ) -> tuple[dict[str, CheckReport], list[Any], int]:
        conc = ctx.reg.policy("run.concurrency.ontology_checks")
        ontology = draft["ontology"]
        checks: dict[str, CheckReport] = {}
        decisions: list[Any] = []
        # The drafted scope decides what S1 triage keeps, so Jev checks it against the whole sample.
        scope = {"name": ctx.reg.corpus["name"], "definition": draft["corpus"]["scope"], "evidence": sorted(texts)}
        checks[SCOPE_KIND] = await check_definitions(ctx.jev, SCOPE_KIND, [scope], texts, conc)
        decisions.extend(checks[SCOPE_KIND].decisions)
        for kind in ONTOLOGY_KINDS:
            if ontology.get(kind):
                checks[kind] = await check_definitions(ctx.jev, kind, ontology[kind], texts, conc)
                decisions.extend(checks[kind].decisions)
        rounds = 0
        while RELATIONS in checks and checks[RELATIONS].overlaps and rounds < ctx.reg.policy("bootstrap.max_disambiguation_rounds"):
            rounds += 1
            result, _ = await ctx.jobs.run_single(
                PROMPT_DISAMBIGUATE, SCHEMA_DISAMBIGUATE, profile,
                {"relations": ontology[RELATIONS], "overlaps": checks[RELATIONS].overlaps, "passages": texts}, ws / f"round{rounds}")
            if not result.ok or not isinstance(result.structured, dict):
                break
            ontology[RELATIONS] = result.structured["relations"]
            checks[RELATIONS] = await check_definitions(ctx.jev, RELATIONS, ontology[RELATIONS], texts, conc,
                                                        overlap_repeat=rounds)  # each round began with flagged overlaps
            decisions.extend(checks[RELATIONS].decisions)
        return checks, decisions, rounds

    async def _check_metadata_fields(self, ctx: Context, draft: dict[str, Any], passages: dict[str, dict[str, Any]]
                                     ) -> tuple[dict[str, list[str]], list[Any]]:
        """Jev keeps a proposed metadata key only if it changes what documents mean; the rest are dropped
        from the draft. A key no sampled document has is dropped without asking."""
        ingest = (draft.get("policies") or {}).get("ingest") or {}
        proposed = ingest.get("metadata_fields") or []
        values: dict[str, set[str]] = defaultdict(set)
        for p in passages.values():
            for k, v in (p.get("metadata") or {}).items():
                values[k].add(v if isinstance(v, str) else str(v))
        kept, dropped, decisions = [], [], []
        shown = ctx.reg.policy("bootstrap.metadata_values")
        for name in dict.fromkeys(proposed):
            if name not in values:
                dropped.append(name)
                continue
            r = await ctx.jev.ask(QS_METADATA, {"corpus": {"scope": draft["corpus"]["scope"]},
                                                "field": {"name": name, "values": sorted(values[name])[:shown]}},
                                  "metadata_field", name)
            decisions.append(r)
            (kept if r.single.outcome == ACCEPT else dropped).append(name)
        if proposed:
            ingest["metadata_fields"] = kept
        return {"kept": kept, "dropped": dropped}, decisions

    async def _check_chrome(self, ctx: Context, draft: dict[str, Any]) -> tuple[dict[str, list[str]], list[Any]]:
        """Corpus-specific page chrome: an element whose exact text recurs on `min_pages` sampled pages is a
        candidate (selector from the page itself), and Jev keeps it only if all it holds is interface
        text; otherwise it stays content."""
        pol = ctx.reg.policy("bootstrap.chrome")
        found = repeated_elements(self.html_pages, ctx.reg.policy("ingest.html"), pol["text_chars"])

        def recurring(sel: str) -> int:
            return max(len(p) for p in found[sel]["texts"].values())

        candidates = sorted((s for s in found if recurring(s) >= pol["min_pages"]),
                            key=lambda s: (-recurring(s), -len(found[s]["pages"]), s))[: pol["max_candidates"]]
        kept, dropped, decisions = [], [], []
        for sel in candidates:
            # the recurring texts and the one-page texts stripping this selector would also remove
            by_pages = sorted(found[sel]["texts"].items(), key=lambda kv: (-len(kv[1]), kv[0]))
            once = [t for t, pages in by_pages if len(pages) == 1]
            texts = list(dict.fromkeys([*(t for t, _ in by_pages[: pol["texts_shown"]]), *once[: pol["texts_shown"]]]))
            r = await ctx.jev.ask(QS_CHROME, {"corpus": {"scope": draft["corpus"]["scope"]},
                                              "element": {"selector": sel, "texts": texts}}, "html_selector", sel)
            decisions.append(r)
            (kept if r.single.outcome == ACCEPT else dropped).append(sel)
        if candidates:
            draft.setdefault("policies", {}).setdefault("ingest", {}).setdefault("html", {})["strip_selectors"] = kept
        return {"kept": kept, "dropped": dropped}, decisions

    def _stage(self, ctx: Context, draft: dict[str, Any], passages: dict[str, dict[str, Any]]) -> str:
        reg: Registry = ctx.reg
        scratch = ctx.jobs.root / self.name / "draft"
        root = start_draft(reg.root, scratch)
        for kind in ONTOLOGY_KINDS:
            items = draft["ontology"].get(kind)
            if items:
                items = [{**i, "evidence": _where(i.get("evidence", []), passages)} for i in items]
                (root / "ontology" / f"{kind}.yaml").write_text(
                    yaml.safe_dump({"items": items}, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if draft.get("policies"):
            path = root / "policies.yaml"
            current = yaml.safe_load(path.read_text(encoding="utf-8"))
            path.write_text(yaml.safe_dump(_merge(current, draft["policies"]), sort_keys=False, allow_unicode=True), encoding="utf-8")
        corpus_path = root / "corpus.yaml"
        corpus = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
        corpus.update({"scope": draft["corpus"]["scope"], "languages": draft["corpus"]["languages"]})
        corpus_path.write_text(yaml.safe_dump(corpus, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if draft.get("extraction_notes"):
            (root / "ontology" / "extraction_notes.md").write_text(draft["extraction_notes"], encoding="utf-8")
        return stage_proposal(reg.root, root).version


def _merge(base: Any, over: Any) -> Any:
    """Only keys that already exist are overridden: the harness tunes values, it
    does not invent policy structure."""
    if isinstance(base, dict) and isinstance(over, dict):
        return {k: _merge(v, over[k]) if k in over else v for k, v in base.items()}
    if isinstance(base, (int, float)) and isinstance(over, (int, float)) and not isinstance(base, bool):
        return over if math.isfinite(over) else base
    return over if type(over) is type(base) else base


def _where(evidence: list[Any], passages: dict[str, dict[str, Any]]) -> list[Any]:
    """Cited sample IDs mean nothing outside this run; keep where each cited passage came from.
    A relation cites (source, target) pairs, each end resolved the same way."""
    def at(c: str) -> dict[str, Any]:
        return {"path": passages[c]["path"], "unit": passages[c]["unit"]}

    out: list[Any] = []
    for e in evidence:
        if isinstance(e, dict):
            if e.get("source") in passages and e.get("target") in passages:
                out.append({"source": at(e["source"]), "target": at(e["target"])})
        elif e in passages:
            out.append(at(e))
    return out
