"""Question-set authoring (§7.3 steps 1, 2 and 7).

A harness job (prompt `author_question_set`, with the TypeSafe skills plugin, R-013)
drafts a question set from the ontology, the decision to make, sampled states and
the jaggedness rules. Code lints the draft (§7.4) and hands lint errors back to the harness for up to
`policies.qs_authoring.max_rounds` rounds. The result is staged as a Registry
proposal; the human gate follows (`registry approve`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jev_graph_builder import log
from jev_graph_builder.ids import unit_fraction
from jev_graph_builder.registry.lint import lint_question_set
from jev_graph_builder.registry.loader import ONTOLOGY_KINDS, Registry, RegistryError
from jev_graph_builder.registry.versioning import Proposal, stage_proposal, start_draft
from jev_graph_builder.store.db import Database

PROMPT = "author_question_set"
SCHEMA = "question_set_draft"
QUESTION_SET_DIR = "question_sets"


@dataclass
class AuthorReport:
    name: str
    version: int
    rounds: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    proposal: Proposal | None = None
    harness_run_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.proposal is not None and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {"question_set": f"{self.name}@{self.version}", "rounds": self.rounds, "ok": self.ok,
                "errors": self.errors, "warnings": self.warnings, "harness_run_ids": self.harness_run_ids,
                "proposal": self.proposal.version if self.proposal else None}


class QuestionSetAuthor:
    def __init__(self, reg: Registry, db: Database, jobs: Any, corpus_id: str) -> None:
        self.reg = reg
        self.db = db
        self.jobs = jobs
        self.corpus_id = corpus_id
        self.policy = reg.policy("qs_authoring")

    def _current(self, name: str) -> dict[str, Any] | None:
        try:
            return self.reg.question_set(name).raw
        except RegistryError:
            return None

    def next_version(self, name: str) -> int:
        versions = [qs.version for qs in self.reg.all_question_sets() if qs.name == name]
        return max(versions, default=0) + 1

    async def samples(self, name: str) -> tuple[list[Any], list[str]]:
        """States this question set (any version) was asked on, and corpus passages."""
        states = await self.db.fetch(
            "SELECT DISTINCT ON (state_hash) state_hash, state FROM jev_calls WHERE question_set = %s AND state IS NOT NULL "
            "ORDER BY state_hash, qs_version DESC", (name,))
        states.sort(key=lambda r: unit_fraction(self.corpus_id, name, r["state_hash"]))
        chunks = await self.db.fetch(
            "SELECT c.chunk_id, c.text FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE d.corpus_id = %s", (self.corpus_id,))
        chunks.sort(key=lambda r: unit_fraction(self.corpus_id, name, r["chunk_id"]))
        return ([r["state"] for r in states[: self.policy["sample_states"]]],
                [r["text"] for r in chunks[: self.policy["sample_passages"]]])

    def lint(self, doc: dict[str, Any], scratch: Path) -> tuple[list[str], list[str], Path]:
        root = start_draft(self.reg.root, scratch)
        path = root / QUESTION_SET_DIR / f"{doc['name']}@{doc['version']}.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
        try:
            draft_reg = Registry(root)
            qs = draft_reg.question_set(doc["name"], doc["version"])
        except RegistryError as e:
            return [str(e)], [], root
        limits = draft_reg.profile("jev", draft_reg.policy("jev.limits_profile"))["limits"]
        report = lint_question_set(draft_reg, qs, limits)
        return report.errors, report.warnings, root

    async def draft(self, name: str, decision: str | None = None, feedback: list[Any] | None = None) -> AuthorReport:
        current = self._current(name)
        purpose = decision or (current or {}).get("purpose")
        if not purpose:
            raise RegistryError(f"question set `{name}` does not exist; pass the decision it should make")
        version = self.next_version(name)
        report = AuthorReport(name, version)
        states, passages = await self.samples(name)
        profile = await self.jobs.choose_profile(PROMPT, {"records": [purpose]})
        ontology = {k: self.reg.ontology(k) for k in ONTOLOGY_KINDS}
        previous, errors = current, list(feedback or [])
        for round_no in range(1, self.policy["max_rounds"] + 1):
            report.rounds = round_no
            ws = self.jobs.root / PROMPT / f"{name}@{version}" / f"round-{round_no}"
            result, run_id = await self.jobs.run_single(PROMPT, SCHEMA, profile, {
                "name": name, "version": version, "decision": purpose, "ontology": ontology,
                "states": states, "passages": passages, "previous": previous, "errors": errors,
                "policy_refs": sorted(self.reg.policies), "question_set_names": self.reg.question_set_names(),
            }, ws)
            report.harness_run_ids.append(run_id)
            if not result.ok or not isinstance(result.structured, dict):
                errors = report.errors = [f"harness run failed: {result.error}"]
                continue
            doc = {**result.structured["question_set"], "name": name, "version": version}
            errors, warnings, root = self.lint(doc, ws / "registry")
            report.errors, report.warnings = errors, warnings
            if not errors:
                report.proposal = stage_proposal(self.reg.root, root)
                break
            previous = doc
            log.get().info("qs_draft_lint_failed", question_set=f"{name}@{version}", round=round_no, errors=len(errors))
        return report
