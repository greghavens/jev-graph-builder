"""Resumable harness batch jobs (§8.4 step 1, §12.2, §14.4).

A job packs records into `input.jsonl` in its own workspace. The prompt tells
the harness to read that file and write one JSON object per input record to
`output.jsonl`. This module:

  * validates each output record against the prompt's `record_schema`,
    keeping the valid ones (partial output survives a crash);
  * resumes the session once with the Registry's continue message for records
    that are still missing or invalid, and otherwise restarts with only those;
  * records every harness run in `harness_runs`, keyed by the job so a crashed
    job resumes instead of re-running completed work;
  * chooses the harness profile with `QS.harness_route`, unless policy pins one
    (R-100, P7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import jsonschema

from jev_graph_builder import log
from jev_graph_builder.harness.base import HarnessAdapter, HarnessError, RunResult, RunSpec
from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.registry.loader import Registry
from jev_graph_builder.store import repo
from jev_graph_builder.store.db import Database

INPUT_FILE = "input.jsonl"
OUTPUT_FILE = "output.jsonl"
KEPT_FILE = "output.kept.jsonl"
RECORD_ID = "id"
WORKSPACE_KEY_CHARS = 16  # hash prefix naming a job workspace directory


@dataclass
class JobOutcome:
    job_key: str
    profile: str
    harness: str
    records: dict[str, dict[str, Any]]           # id -> valid record
    missing: dict[str, str]                      # id -> reason
    harness_run_ids: list[str] = field(default_factory=list)
    model: str | None = None


def read_jsonl(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append(None)
    return out


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class JobRunner:
    def __init__(
        self,
        registry: Registry,
        adapters: dict[str, HarnessAdapter],
        db: Database | None,
        workspace_root: Path,
        run_id: str | None = None,
        router: Callable[[str, dict[str, Any]], Any] | None = None,
        harness_override: str | None = None,
    ) -> None:
        self.reg = registry
        self.adapters = adapters
        self.db = db
        self.root = Path(workspace_root)
        self.run_id = run_id
        self.router = router
        self.harness_override = harness_override

    # -------------------------------------------------------------- profiles

    def workspace_for(self, job_name: str, key: str) -> Path:
        """Deterministic workspace directory for a job keyed by a content hash."""
        return self.root / job_name / key[:WORKSPACE_KEY_CHARS]

    async def choose_profile(self, job_name: str, route_state: dict[str, Any], exclude: set[str] | None = None) -> str:
        """Pinned by policy, else overridden per command, else `QS.harness_route` (R-100)."""
        pinned = self.reg.policy_or(f"harness.pinned.{job_name}", None)
        if pinned and pinned not in (exclude or set()):
            return self._apply_override(pinned)
        if self.router is not None:
            profile = await self.router(job_name, {**route_state, "exclude": sorted(exclude or ())})
            if profile:
                return self._apply_override(profile)
        default = self.reg.policy(f"harness.defaults.{job_name}")
        return self._apply_override(default)

    def _apply_override(self, profile: str) -> str:
        """`--harness claude_code|codex` swaps to the same-tier profile of that harness."""
        if not self.harness_override or self.harness_override == "auto":
            return profile
        prof = self.reg.profile("harness", profile)
        if prof["harness"] == self.harness_override:
            return profile
        twin = prof.get("twins", {}).get(self.harness_override)
        if not twin:
            raise HarnessError(f"profile `{profile}` has no {self.harness_override} twin for --harness")
        return twin

    def other_harness_profile(self, profile: str) -> str | None:
        prof = self.reg.profile("harness", profile)
        other = "codex" if prof["harness"] == "claude_code" else "claude_code"
        return prof.get("twins", {}).get(other)

    # ------------------------------------------------------------------ jobs

    def job_key(self, job_name: str, prompt_ref: str, profile: str, records: list[dict[str, Any]]) -> str:
        prompt = self.reg.prompt(prompt_ref)
        schema_ref = prompt.meta.get("record_schema") or prompt.meta["output_schema"]
        return sha256_hex(
            job_name, prompt.ref, prompt.content_hash, self.reg.artifact_hash(f"schemas/{schema_ref}.json"),
            profile, self.reg.profile("harness", profile), [sha256_hex(r) for r in records],
        )

    async def run_job(
        self,
        job_name: str,
        prompt_ref: str,
        records: list[dict[str, Any]],
        profile: str,
        variables: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> JobOutcome:
        if not records:
            return JobOutcome("", profile, self.reg.profile("harness", profile)["harness"], {}, {})
        ids = [str(r[RECORD_ID]) for r in records]
        if len(set(ids)) != len(ids):
            raise HarnessError(f"{job_name}: duplicate record ids")
        prompt = self.reg.prompt(prompt_ref)
        record_schema = self.reg.schema(prompt.meta.get("record_schema") or prompt.meta["output_schema"])
        key = self.job_key(job_name, prompt_ref, profile, records)
        ws = self.workspace_for(job_name, key)
        ws.mkdir(parents=True, exist_ok=True)
        for name, content in (files or {}).items():
            (ws / name).write_text(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8")

        prof = self.reg.profile("harness", profile)
        adapter = self.adapters[prof["harness"]]
        outcome = JobOutcome(key, profile, prof["harness"], {}, {}, model=prof.get("model"))
        by_id = {str(r[RECORD_ID]): r for r in records}

        self._collect(ws / KEPT_FILE, by_id, record_schema, outcome)
        self._collect(ws / OUTPUT_FILE, by_id, record_schema, outcome)
        pending = [i for i in ids if i not in outcome.records]
        attempts_left = self.reg.policy("harness.job_attempts")
        session = await self._last_session(key)

        while pending and attempts_left > 0:
            attempts_left -= 1
            write_jsonl(ws / KEPT_FILE, [outcome.records[i] for i in ids if i in outcome.records])
            spec_vars = {
                **(variables or {}),
                "input_file": INPUT_FILE, "output_file": OUTPUT_FILE, "record_count": len(pending),
                "record_id_field": RECORD_ID,
            }
            spec = RunSpec(prompt_ref=prompt.ref, variables=spec_vars, output_schema_ref=None, workspace=ws, profile=profile)
            if session is not None:
                # §14.4: resume the crashed / incomplete session and ask for the rest.
                message = self._jinja_render(self.reg.policy("harness.continue_message_ref"), missing_ids=pending, output_file=OUTPUT_FILE)
                result = await self._tracked(adapter, key, spec, lambda: adapter.continue_with(session, spec, message))  # type: ignore[attr-defined]
                session = None
            else:
                write_jsonl(ws / INPUT_FILE, [by_id[i] for i in pending])
                (ws / OUTPUT_FILE).unlink(missing_ok=True)
                result = await self._tracked(adapter, key, spec, lambda: adapter.run(spec))
                session = result.session_id if result.session_id and not result.extra.get("timed_out") else None
            outcome.harness_run_ids.append(result.extra["harness_run_id"])
            self._collect(ws / OUTPUT_FILE, by_id, record_schema, outcome)
            pending = [i for i in ids if i not in outcome.records]
            if not pending:
                break
        write_jsonl(ws / KEPT_FILE, [outcome.records[i] for i in ids if i in outcome.records])
        for i in pending:
            outcome.missing.setdefault(i, "no valid record")
        for i in outcome.records:
            outcome.missing.pop(i, None)
        return outcome

    # --------------------------------------------------------------- helpers

    def _jinja_render(self, prompt_ref: str, **variables: Any) -> str:
        import jinja2

        body = self.reg.prompt(prompt_ref).body
        return jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False).from_string(body).render(**variables)

    def _collect(self, path: Path, by_id: dict[str, dict], schema: dict, outcome: JobOutcome) -> None:
        for row in read_jsonl(path):
            if not isinstance(row, dict) or str(row.get(RECORD_ID)) not in by_id:
                continue
            rid = str(row[RECORD_ID])
            try:
                jsonschema.validate(row, schema)
            except jsonschema.ValidationError as exc:
                outcome.missing[rid] = f"schema: {exc.message}"
                continue
            outcome.records[rid] = row

    async def _last_session(self, job_key: str) -> str | None:
        if self.db is None:
            return None
        row = await self.db.fetchone(
            "SELECT session_id FROM harness_runs WHERE job_key = %s AND session_id IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            (job_key,),
        )
        return row["session_id"] if row else None

    async def _tracked(self, adapter: HarnessAdapter, job_key: str, spec: RunSpec, call: Callable) -> RunResult:
        prof = self.reg.profile("harness", spec.profile)
        started = datetime.now(UTC)
        run_row_id = sha256_hex(job_key, started.isoformat())
        base = {
            "harness_run_id": run_row_id, "harness": prof["harness"], "profile": spec.profile, "model": prof.get("model"),
            "prompt_ref": spec.prompt_ref, "schema_ref": spec.output_schema_ref, "workspace": str(spec.workspace),
            "started_at": started, "run_id": self.run_id, "job_key": job_key,
        }
        if self.db is not None:
            async with self.db.tx() as conn:
                await repo.upsert(conn, "harness_runs", base, key=("harness_run_id",))
        log.get().info("harness_run_started", harness_run_id=run_row_id, harness=prof["harness"], profile=spec.profile,
                       prompt_ref=spec.prompt_ref, workspace=str(spec.workspace))
        try:
            result = await call()
        except Exception as exc:
            if self.db is not None:
                async with self.db.tx() as conn:
                    await repo.upsert(conn, "harness_runs", {**base, "ok": False, "error": str(exc)[:2000], "ended_at": datetime.now(UTC)}, key=("harness_run_id",))
            raise
        if self.db is not None:
            async with self.db.tx() as conn:
                await repo.upsert(
                    conn, "harness_runs",
                    {**base, "session_id": result.session_id, "ok": result.ok, "error": result.error, "usage": result.usage,
                     "events_path": str(result.events_path), "ended_at": datetime.now(UTC)},
                    key=("harness_run_id",),
                )
        log.get().info("harness_run", harness_run_id=run_row_id, harness=result.harness, profile=spec.profile,
                       ok=result.ok, error=result.error)
        result.extra["harness_run_id"] = run_row_id
        return result

    async def run_single(self, prompt_ref: str, schema_ref: str, profile: str, variables: dict[str, Any], workspace: Path) -> tuple[RunResult, str]:
        """One structured call (e.g. the Graph RAG answer, adjudication). Returns (result, harness_run_id)."""
        prof = self.reg.profile("harness", profile)
        adapter = self.adapters[prof["harness"]]
        spec = RunSpec(prompt_ref=prompt_ref, variables=variables, output_schema_ref=schema_ref, workspace=workspace, profile=profile)
        key = sha256_hex(prompt_ref, schema_ref, profile, variables)
        result = await self._tracked(adapter, key, spec, lambda: adapter.run(spec))
        return result, result.extra["harness_run_id"]
