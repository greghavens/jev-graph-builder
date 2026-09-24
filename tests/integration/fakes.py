"""Deterministic fakes for integration tests: Jev (HTTP transport), harness adapters, embeddings.

The fake Jev recovers `(question set, logical question)` from the rendered
instructions (keys are opaque), then answers from a small table. Answers are
confident, so gating takes the accept/reject paths rather than review.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import httpx2

from jev_graph_builder.harness.base import RunResult, RunSpec
from jev_graph_builder.registry.loader import Registry

YES, NO, SURE = 0.97, 0.02, 0.99

# Questions whose "true" answer is the unusual case in a clean corpus.
NO_QUESTIONS = {
    ("injection", "contains_injection"), ("doc_triage", "injection"), ("rag_passage", "injection"),
    ("lint_question", "forbidden_task"), ("link", "contradiction"), ("link_fanout", "contradiction"),
    ("dedup", "same_question"), ("relation_overlap", "overlap"),
    ("audit_orphan", "should_connect"), ("chunk_classify", "boilerplate"),
    ("train_negative", "answers"), ("train_multihop", "answerable"), ("rag_passage", "contradicts"),
    ("entity_align", "same_referent"),
    *((qs, q) for qs in ("extract_verify", "extract_verify_fanout")
      for q in ("name_wrong", "span_wrong", "attributes_wrong", "text_wrong", "evidence_wrong", "type_wrong")),
    ("chunk_check_fanout", "contains_injection"), ("chunk_check_fanout", "boilerplate"),
    ("metadata_derivable", "per_document"),
    *((qs, q) for qs in ("train_qa", "train_instruction") for q in ("leaks_answer", "contains_pii")),
}
# Scores whose clean-corpus answer is the lowest level (default: the highest).
LOW_SCORES = {("entity_align", "alignment"), ("query_route", "complexity")}
NOT_AN_OPTION = {"none", "neither", "other"}


def _pattern(instructions: str) -> re.Pattern[str]:
    parts = re.split(r"\{\{\s*item\s*\}\}", instructions)
    return re.compile("^" + ".*?".join(re.escape(p) for p in parts) + "$", re.S)


class FakeJev:
    def __init__(self, reg: Registry, overrides: dict[tuple[str, str], Any] | None = None) -> None:
        self.index: list[tuple[re.Pattern[str], str, str]] = []
        self.sizes: dict[str, int] = {}
        for qs in reg.all_question_sets():
            self.sizes[qs.name] = len(qs.raw["questions"])
            for logical, q in qs.raw["questions"].items():
                self.index.append((_pattern(q["instructions"]), qs.name, logical))
        self.overrides = overrides or {}
        self.calls: list[dict[str, Any]] = []
        self.fail_next: int = 0  # simulate an outage for N calls

    def candidates(self, instructions: str) -> dict[str, str]:
        found = {qs: logical for pat, qs, logical in self.index if pat.match(instructions)}
        if not found:
            raise AssertionError(f"fake Jev: unknown question {instructions[:80]!r}")
        return found

    def identify_call(self, questions: dict[str, Any]) -> dict[str, tuple[str, str]]:
        """Identical wording may appear in several sets (e.g. `rag_passage.answers` and
        `train_negative.answers`); the call's other questions pick the set: the smallest
        set that covers every question in the call."""
        per_key = {k: self.candidates(q["instructions"]) for k, q in questions.items()}
        common = set.intersection(*(set(c) for c in per_key.values()))
        if not common:
            return {k: next(iter(c.items())) for k, c in per_key.items()}
        qs = min(sorted(common), key=lambda n: self.sizes[n])
        return {k: (qs, c[qs]) for k, c in per_key.items()}

    def answer(self, body: dict[str, Any], ident: tuple[str, str]) -> dict[str, Any]:
        qs, logical = ident
        override = self.overrides.get((qs, logical))
        t = body["type"]
        if t == "noul":
            p = override if override is not None else (NO if (qs, logical) in NO_QUESTIONS else YES)
            return {"type": "noul", "noul": p}
        if t == "choice":
            opts = list(body["criteria"])
            pick = override if override is not None else next((o for o in opts if o not in NOT_AN_OPTION), opts[0])
            return {"type": "choice", "choice": pick, "confidence": SURE,
                    "probabilities": {o: (SURE if o == pick else (1 - SURE) / max(len(opts) - 1, 1)) for o in opts}}
        levels = len(body["criteria"])
        top = override if override is not None else (0 if (qs, logical) in LOW_SCORES else levels - 1)
        return {"type": "score", "score": float(top), "confidence": SURE,
                "legend": {str(i): lvl for i, lvl in enumerate(body["criteria"])},
                "probabilities": {str(i): (1.0 if i == top else 0.0) for i in range(levels)}}

    def transport(self) -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if self.fail_next > 0:
                self.fail_next -= 1
                return httpx2.Response(503, json={"error": {"message": "unavailable"}})
            payload = json.loads(request.content)
            self.calls.append(payload)
            ident = self.identify_call(payload["questions"])
            answers = {k: self.answer(q, ident[k]) for k, q in payload["questions"].items()}
            chars = len(json.dumps(payload))
            return httpx2.Response(200, json={"model": payload["model"], "answers": answers,
                                              "usage": {"input_tokens": chars // 4, "output_tokens": len(answers) * 8}},
                                   headers={"x-typesafe-request-id": "req_" + uuid.uuid4().hex})

        return httpx2.MockTransport(handler)


# -- harness -----------------------------------------------------------------

SENT = re.compile(r"[^.!?\n]+[.!?]")
KNOWN_ENTITIES = {"Ledgerd": "service", "Courier": "service", "Atlas": "service", "Quill": "service", "Sentinel": "service",
                  "Postgres": "datastore", "Kafka": "datastore", "etcd": "datastore", "Redis": "datastore",
                  "Prometheus": "service"}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _sentence(record: Any) -> str:
    for s in _strings(record):
        m = SENT.search(s)
        if m and len(m.group(0).strip()) > 20:
            return m.group(0).strip()
    return next((s for s in _strings(record) if s.strip()), "n/a")


def _ids(value: Any, key: str = "id") -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k in (key, "chunk_id") and isinstance(v, (str, int)):
                out.append(str(v))
            out += _ids(v, key)
    elif isinstance(value, list):
        for v in value:
            out += _ids(v, key)
    return out


def fake_record(prompt: str, rec: dict[str, Any]) -> dict[str, Any]:
    text = rec.get("text") or ""
    sentence = _sentence(rec)
    if prompt == "extract":
        ents = [{"name": n, "type": t, "span": n, "attributes": {"kind": t}} for n, t in KNOWN_ENTITIES.items() if n in text]
        claims = [{"text": sentence, "claim_type": "behavior", "evidence_span": sentence}] if sentence in text else []
        return {"id": rec["id"], "entities": ents, "claims": claims, "summary": sentence,
                "title": text.splitlines()[0][:60] if text else "untitled", "keywords": [e["name"] for e in ents][:3]}
    if prompt == "gen_qa_single":
        return {"id": rec["id"], "question": f"What does the passage say about {sentence.split()[0]}?", "answer": sentence, "evidence": sentence}
    if prompt == "gen_instruction":
        return {"id": rec["id"], "instruction": "Summarize the operational rule in this passage.", "response": sentence, "evidence": sentence}
    if prompt == "gen_qa_multihop":
        hops = [{"chunk_id": p["id"], "span": _sentence(p) if p.get("text") else sentence} for p in rec.get("passages", [])]
        return {"id": rec["id"], "question": "How are these two facts connected?", "answer": sentence, "hop_evidence": hops}
    if prompt == "gen_graph_reasoning":
        return {"id": rec["id"], "question": "Which components are connected here?", "answer": sentence}
    raise AssertionError(f"fake harness: no record generator for {prompt}")


def fake_structured(prompt: str, variables: dict[str, Any]) -> dict[str, Any]:
    if prompt == "answer":
        ids = _ids(variables.get("passages"))
        text = _sentence(variables.get("passages"))
        return {"answer_sentences": [{"text": text, "citation_ids": ids[:1]}]} if ids else {"answer_sentences": []}
    if prompt == "community_summary":
        ids = _ids(variables.get("passages"))
        return {"sentences": [{"text": _sentence(variables.get("passages")), "citation_ids": ids[:1]}]}
    if prompt in ("propose_relations", "disambiguate_relations"):
        return {"relations": []}
    if prompt == "bootstrap_registry":
        # Keeps the current ontology, adds one entity type grounded in the fixture text; every item
        # cites the first sample passage, as the draft schema requires.
        cite = [p["id"] for p in (variables.get("passages") or [])][:1]
        allowed = ("name", "definition", "direction", "direction_semantics", "allowed_src", "allowed_dst", "parent")

        def draft(kind: str, item: dict[str, Any]) -> dict[str, Any]:
            out = {k: v for k, v in item.items() if k in allowed}
            if kind == "relation_types":
                out.setdefault("direction_semantics", "Either end may be the source.")
            evidence = [{"source": c, "target": c} for c in cite] if kind == "relation_types" else cite
            return {**out, "examples": [item["definition"]], "evidence": evidence}

        ontology = {kind: [draft(kind, item) for item in items]
                    for kind, items in (variables.get("current") or {}).items() if items}
        ontology.setdefault("entity_types", []).append(draft("entity_types", {
            "name": "config_repository", "definition": "A shared repository that holds service configuration."}))
        corpus = variables.get("corpus") or {}
        scope = corpus.get("scope") or "Technical documentation for a software platform: services, pipelines and incidents."
        return {"ontology": ontology, "corpus": {"scope": scope, "languages": ["en"]},
                "open_decisions": ["Should incident reviews link to the runbook they cite?"]}
    raise AssertionError(f"fake harness: no structured output for {prompt}")


class FakeHarness:
    """Stands in for both Claude Code and Codex; records every run."""

    def __init__(self, reg: Registry, name: str = "claude_code", crash_after_records: int | None = None) -> None:
        self.reg = reg
        self.name = name
        self.runs: list[RunSpec] = []
        self.crash_after_records = crash_after_records
        self.relation_proposal: dict[str, Any] | None = None  # set: propose_relations proposes it, citing every example

    async def run(self, spec: RunSpec) -> RunResult:
        self.runs.append(spec)
        prompt = self.reg.prompt(spec.prompt_ref)
        sid = str(uuid.uuid4())
        events = spec.workspace.parent / f"{spec.workspace.name}.runs" / sid / "events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        events.write_text("")
        if prompt.meta.get("output_mode") == "file":
            inp = spec.workspace / spec.variables["input_file"]
            records = [json.loads(line) for line in inp.read_text().splitlines() if line.strip()]
            if self.crash_after_records is not None:
                records = records[: self.crash_after_records]
            out = spec.workspace / spec.variables["output_file"]
            out.write_text("".join(json.dumps(fake_record(prompt.name, r)) + "\n" for r in records))
            return RunResult(self.name, sid, True, None, "done", None, [out], {"input_tokens": 1000}, events, "fake")
        if prompt.name == "propose_relations" and self.relation_proposal is not None:
            ids = [e["id"] for e in spec.variables.get("examples") or []]
            structured = {"relations": [{**self.relation_proposal, "evidence": ids}] if ids else []}
        else:
            structured = fake_structured(prompt.name, spec.variables)
        return RunResult(self.name, sid, True, None, json.dumps(structured), structured, [], {"input_tokens": 500}, events, "fake")

    async def resume(self, session_id: str, spec: RunSpec, fork: bool = False) -> RunResult:
        return await self.run(spec)

    async def continue_with(self, session_id: str, spec: RunSpec, message: str) -> RunResult:
        return await self.run(spec)


class FakeEmbedder:
    """Hash-seeded unit vectors: identical text -> identical vector; shared words -> nearby vectors."""

    def __init__(self, profile: dict[str, Any]) -> None:
        self.model_id = "fake/" + profile["model_id"]
        self.dim = profile["dim"]
        self.max_tokens = profile["max_tokens"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


def write_docs(dst: Path, src: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(src.glob("*.md")):
        q = dst / p.name
        q.write_text(p.read_text())
        out.append(q)
    return out
