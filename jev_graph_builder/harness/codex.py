"""Codex adapter (§12.2).

`codex exec --json --output-schema <file> -o <file> -s <profile sandbox>
 -c approval_policy="never" -m <model> -C <workspace> --skip-git-repo-check -`

The prompt is read from stdin (`-`). The session ID is `thread.started.thread_id`.
Codex does not enforce the schema, so this adapter validates the final message
itself and asks for one repair with `codex exec resume`. `exec resume` has no `-s`/`-C` flags, so the
sandbox is passed as `-c sandbox_mode=…` and the workspace as the process cwd.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from jev_graph_builder.harness.base import EVENTS_FILE, HarnessBase, RunResult, RunSpec
from jev_graph_builder.harness.claude import session_dir

THREAD_STARTED = "thread.started"
TURN_COMPLETED = "turn.completed"
TURN_FAILED = "turn.failed"
ITEM_COMPLETED = "item.completed"
AGENT_MESSAGE = "agent_message"
ERROR_EVENT = "error"


def _toml_str(value: str) -> str:
    return json.dumps(value)


class CodexAdapter(HarnessBase):
    name = "codex"

    def _common(self, prof: dict[str, Any], schema_file: Path | None, last_file: Path) -> list[str]:
        argv = ["--json", "-o", str(last_file), "-m", prof["model"], "--skip-git-repo-check",
                "-c", f"approval_policy={_toml_str(self.reg.policy('harness.codex_approval_policy'))}"]
        if prof.get("reasoning_effort"):
            argv += ["-c", f"model_reasoning_effort={_toml_str(prof['reasoning_effort'])}"]
        if schema_file is not None:
            argv += ["--output-schema", str(schema_file)]
        return argv

    def _files(self, spec: RunSpec, session_key: str) -> tuple[Path, Path | None, Path]:
        rdir = session_dir(spec.workspace, session_key)
        rdir.mkdir(parents=True, exist_ok=True)
        schema = self.schema(spec)
        schema_file = None
        if schema is not None:
            schema_file = rdir / "output_schema.json"
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
        return rdir / EVENTS_FILE, schema_file, rdir / "last_message.txt"

    def _install_instructions(self, spec: RunSpec) -> None:
        """R-013 for Codex: Codex has no Claude plugins; a plugin's instruction file is
        placed in the workspace as `AGENTS.md`, which Codex reads."""
        texts = [Path(p["codex_instructions"]).read_text(encoding="utf-8") for p in self.plugins(spec) if p.get("codex_instructions")]
        if texts:
            (spec.workspace / "AGENTS.md").write_text("\n\n".join(texts), encoding="utf-8")

    async def _execute(self, spec: RunSpec, resume: str | None, fork: bool, prompt: str | None = None) -> RunResult:
        prof = self.profile(spec.profile)
        spec.workspace.mkdir(parents=True, exist_ok=True)
        run_key = resume if (resume and not fork) else self.new_session_id()
        events_path, schema_file, last_file = self._files(spec, run_key)
        common = self._common(prof, schema_file, last_file)
        if resume is None:
            argv = [self.binary, "exec", *common, "-s", prof["sandbox"], "-C", str(spec.workspace), "-"]
        else:
            sub = "fork" if fork else "resume"
            argv = [self.binary, "exec", sub, *common, "-c", f"sandbox_mode={_toml_str(prof['sandbox'])}", resume, "-"]
        self._install_instructions(spec)
        before = {p for p in spec.workspace.rglob("*") if p.is_file()}
        out = await self.spawn(argv, cwd=spec.workspace, env=self.env_for(prof), timeout_s=prof["timeout_s"],
                               events_path=events_path, stdin=prompt if prompt is not None else self.render_prompt(spec))
        thread_id = resume if (resume and not fork) else None
        usage: dict[str, Any] = {}
        final_text: str | None = None
        errors: list[str] = []
        for ev in out.events:
            t = ev.get("type")
            if t == THREAD_STARTED:
                thread_id = ev.get("thread_id") or thread_id
            elif t == TURN_COMPLETED:
                for k, v in (ev.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        usage[k] = usage.get(k, 0) + v
            elif t == ITEM_COMPLETED and (ev.get("item") or {}).get("type") == AGENT_MESSAGE:
                final_text = ev["item"].get("text")
            elif t in (TURN_FAILED, ERROR_EVENT):
                errors.append(json.dumps(ev.get("error") or ev.get("message"))[:500])
        if final_text is None and last_file.is_file():
            final_text = last_file.read_text(encoding="utf-8") or None
        files = [p for p in self.output_files(spec.workspace, before)]
        structured, error = None, None
        ok = out.returncode == 0 and not out.timed_out and not errors
        if not ok:
            error = "timeout" if out.timed_out else ("; ".join(errors) or f"exit {out.returncode}: {out.stderr_tail}")
        elif spec.structured:
            structured, error = self.validate(spec, final_text)
            ok = error is None
        return RunResult(
            harness=self.name,
            session_id=thread_id,  # None when Codex never started a thread: nothing to resume
            ok=ok,
            error=error,
            final_text=final_text,
            structured=structured,
            output_files=files,
            usage=usage,
            events_path=events_path,
            model=prof["model"],
            extra={"timed_out": out.timed_out},
        )

    def validate(self, spec: RunSpec, text: str | None) -> tuple[Any, str | None]:
        if not text:
            return None, "no final message"
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"final message is not JSON: {exc}"
        try:
            jsonschema.validate(value, self.schema(spec))
        except jsonschema.ValidationError as exc:
            return None, f"schema: {exc.message}"
        return value, None

    async def run(self, spec: RunSpec) -> RunResult:
        if spec.resume_session_id:
            return await self.resume(spec.resume_session_id, spec, spec.fork)
        result = await self._execute(spec, None, False)
        if not result.ok and spec.structured and result.error and result.session_id and not result.extra.get("timed_out"):
            # One repair turn in the same session (§12.2).
            repair = self.reg.policy("harness.repair_message_ref")
            message = self._jinja.from_string(self.reg.prompt(repair).body).render(error=result.error)
            repaired = await self.continue_with(result.session_id, spec, message)
            repaired.extra["repaired_from_error"] = result.error
            return repaired
        return result

    async def resume(self, session_id: str, spec: RunSpec, fork: bool = False) -> RunResult:
        return await self._execute(spec, session_id, fork)

    async def continue_with(self, session_id: str, spec: RunSpec, message: str) -> RunResult:
        return await self._execute(spec, session_id, False, prompt=message)
