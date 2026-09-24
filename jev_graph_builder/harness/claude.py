"""Claude Code adapter (§12.2).

`claude -p --output-format stream-json --verbose --json-schema <schema>
 --permission-mode <profile> --allowedTools <profile> --max-turns <policy>
 --model <profile> --session-id <uuid> [--bare]`

The prompt is written to stdin (never argv), every stream event is recorded,
`structured_output` comes from the final `result` event, and `success`
without `structured_output` is a failure when a schema was requested.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jev_graph_builder.harness.base import EVENTS_FILE, HarnessBase, HarnessError, HarnessUsageLimit, RunResult, RunSpec

RESULT_EVENT = "result"
SUCCESS = "success"
RATE_LIMIT_EVENT = "rate_limit_event"
REJECTED = "rejected"


def usage_limit(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The CLI's rate-limit event when the account's usage limit rejected the run."""
    for e in events:
        info = e.get("rate_limit_info") or {}
        if e.get("type") == RATE_LIMIT_EVENT and info.get("status") == REJECTED:
            return info
    return None


def session_dir(workspace: Path, session_id: str) -> Path:
    """Run records live beside the workspace, outside the harness's writable area."""
    return workspace.parent / f"{workspace.name}.runs" / session_id


class ClaudeCodeAdapter(HarnessBase):
    name = "claude_code"

    def _argv(self, prof: dict[str, Any], spec: RunSpec, session_id: str, resume: str | None, fork: bool) -> list[str]:
        caps = self.reg.policy("harness.caps")
        argv = [
            self.binary, "-p",
            "--output-format", "stream-json", "--verbose",
            "--permission-mode", prof["permission_mode"],
            "--allowedTools", ",".join(prof.get("allowed_tools") or []),
            "--max-turns", str(prof.get("max_turns", caps["max_turns"])),
            "--model", prof["model"],
        ]
        if prof.get("disallowed_tools"):
            argv += ["--disallowedTools", ",".join(prof["disallowed_tools"])]
        schema = self.schema(spec)
        if schema is not None:
            # The CLI's validator does not resolve the draft meta-schema URI, so the
            # `$schema` declaration is dropped; the schema body is unchanged.
            body = {k: v for k, v in schema.items() if k != "$schema"}
            argv += ["--json-schema", json.dumps(body, separators=(",", ":"))]
        if resume:
            argv += ["--resume", resume]
            if fork:
                argv += ["--fork-session", "--session-id", session_id]
        else:
            argv += ["--session-id", session_id]
        for plugin in self.plugins(spec):
            argv += ["--plugin-dir", str(Path(plugin["dir"]).resolve())] if plugin.get("dir") else ["--plugin-url", plugin["url"]]
        if prof.get("bare"):
            argv.append("--bare")
        if prof.get("extra_args"):
            argv += list(prof["extra_args"])
        return argv

    async def _execute(self, spec: RunSpec, resume: str | None, fork: bool, prompt: str | None = None) -> RunResult:
        prof = self.profile(spec.profile)
        session_id = self.new_session_id() if (resume is None or fork) else resume
        spec.workspace.mkdir(parents=True, exist_ok=True)
        events_path = session_dir(spec.workspace, session_id) / EVENTS_FILE
        before = {p for p in spec.workspace.rglob("*") if p.is_file()}
        out = await self.spawn(
            self._argv(prof, spec, session_id, resume, fork),
            cwd=spec.workspace,
            env=self.env_for(prof),
            timeout_s=prof["timeout_s"],
            events_path=events_path,
            stdin=prompt if prompt is not None else self.render_prompt(spec),
        )
        limit = usage_limit(out.events)
        if limit is not None:
            resets = limit.get("resetsAt")
            when = datetime.fromtimestamp(resets, UTC).isoformat() if isinstance(resets, (int, float)) else "unknown"
            raise HarnessUsageLimit(f"{limit.get('rateLimitType')} usage limit, resets at {when}")
        result = next((e for e in reversed(out.events) if e.get("type") == RESULT_EVENT), None)
        files = self.output_files(spec.workspace, before)
        if result is None:
            error = "timeout" if out.timed_out else f"no result event (exit {out.returncode}): {out.stderr_tail}"
            return RunResult(self.name, session_id, False, error, None, None, files, {}, events_path, prof["model"])
        sid = result.get("session_id") or session_id
        structured = result.get("structured_output")
        ok = result.get("subtype") == SUCCESS and not result.get("is_error")
        error = None if ok else f"{result.get('subtype')}: {str(result.get('result'))[:500]}"
        if ok and spec.structured and structured is None:
            ok, error = False, "success without structured_output"
        return RunResult(
            harness=self.name,
            session_id=sid,
            ok=ok,
            error=error,
            final_text=result.get("result"),
            structured=structured,
            output_files=files,
            usage=result.get("usage") or {},
            events_path=events_path,
            model=prof["model"],
            extra={"num_turns": result.get("num_turns"), "timed_out": out.timed_out},
        )

    async def run(self, spec: RunSpec) -> RunResult:
        if spec.resume_session_id:
            return await self.resume(spec.resume_session_id, spec, spec.fork)
        return await self._execute(spec, None, False)

    async def resume(self, session_id: str, spec: RunSpec, fork: bool = False) -> RunResult:
        return await self._execute(spec, session_id, fork)

    async def continue_with(self, session_id: str, spec: RunSpec, message: str) -> RunResult:
        """Resume with an explicit follow-up message (repair / continue instructions)."""
        return await self._execute(spec, session_id, False, prompt=message)


def ensure_available(binary: str) -> None:
    import shutil

    if shutil.which(binary) is None:
        raise HarnessError(f"`{binary}` not found on PATH")
