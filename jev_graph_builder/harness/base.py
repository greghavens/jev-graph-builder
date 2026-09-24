"""Harness adapter interface (§12.1) and the shared subprocess mechanics.

Security rules enforced here, for every harness:
  * R-020: the child environment is built from an allow-list (policy), after
    dropping every `CLAUDE*` / `CODEX*` session variable; secrets named by the
    profile are added back explicitly.
  * R-021: argv is checked against the forbidden-argument list (policy) before
    any process starts.
  * The workspace is the only writable directory; a stuck process is killed
    after the profile timeout.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import jinja2

from jev_graph_builder.config import secret
from jev_graph_builder.registry.loader import Registry

SESSION_ENV_PREFIXES = ("CLAUDE", "CODEX")
EVENTS_FILE = "events.jsonl"
STDERR_FILE = "stderr.log"


class HarnessError(Exception):
    pass


class ForbiddenArgument(HarnessError):
    """R-021: a permission-bypass flag reached the argv builder."""


class HarnessUsageLimit(HarnessError):
    """The harness account hit its usage limit: every run fails until it resets, so the run stops."""


@dataclass
class RunSpec:
    prompt_ref: str
    variables: dict[str, Any]
    output_schema_ref: str | None
    workspace: Path
    profile: str
    resume_session_id: str | None = None
    fork: bool = False

    def __post_init__(self) -> None:
        # Adapters run the CLI with cwd=workspace and also pass workspace paths as
        # arguments; a relative workspace would resolve against itself.
        self.workspace = Path(self.workspace).resolve()

    @property
    def structured(self) -> bool:
        """The run must return JSON matching its output schema."""
        return self.output_schema_ref is not None


@dataclass
class RunResult:
    harness: Literal["claude_code", "codex"]
    session_id: str | None
    ok: bool
    error: str | None
    final_text: str | None
    structured: Any | None
    output_files: list[Path]
    usage: dict[str, Any]
    events_path: Path
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HarnessAdapter(Protocol):
    name: str

    async def run(self, spec: RunSpec) -> RunResult: ...

    async def resume(self, session_id: str, spec: RunSpec, fork: bool = False) -> RunResult: ...


@dataclass
class ProcOutput:
    returncode: int | None
    events: list[dict[str, Any]]
    timed_out: bool
    stderr_tail: str


def scrubbed_env(allow: list[str], add_secrets: list[str]) -> dict[str, str]:
    """R-020: start from nothing, copy only allow-listed variables, then the named secrets."""
    env: dict[str, str] = {}
    for name in allow:
        if name.startswith(SESSION_ENV_PREFIXES):
            continue
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    for name in add_secrets:
        value = secret(name)
        if value is not None:
            env[name] = value
    return env


def check_argv(argv: list[str], forbidden: list[str]) -> None:
    for arg in argv:
        low = arg.lower()
        for bad in forbidden:
            if bad.lower() in low:
                raise ForbiddenArgument(f"refusing to pass `{arg}` to a harness (R-021)")


class HarnessBase:
    """Shared mechanics: prompt rendering, schema lookup, subprocess with event capture."""

    name: Literal["claude_code", "codex"]

    def __init__(self, registry: Registry, binary: str | None = None) -> None:
        self.reg = registry
        self.binary = binary or registry.policy(f"harness.binaries.{self.name}")
        self.forbidden = registry.policy("harness.forbidden_args")
        self.env_allow = registry.policy("harness.env_allowlist")
        self._jinja = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False, keep_trailing_newline=True)

    # -- helpers ------------------------------------------------------------

    def profile(self, name: str) -> dict[str, Any]:
        prof = self.reg.profile("harness", name)
        if prof["harness"] != self.name:
            raise HarnessError(f"profile `{name}` is for {prof['harness']}, not {self.name}")
        return prof

    def render_prompt(self, spec: RunSpec) -> str:
        prompt = self.reg.prompt(spec.prompt_ref)
        return self._jinja.from_string(prompt.body).render(**spec.variables)

    def schema(self, spec: RunSpec) -> dict[str, Any] | None:
        return self.reg.schema(spec.output_schema_ref) if spec.output_schema_ref else None

    def plugins(self, spec: RunSpec) -> list[dict[str, Any]]:
        """R-013: harness plugins a prompt requires (e.g. the TypeSafe question-writing skills),
        resolved from `policies.harness.plugins.<name>`."""
        names = self.reg.prompt(spec.prompt_ref).meta.get("plugins") or []
        return [self.reg.policy(f"harness.plugins.{n}") for n in names]

    def env_for(self, prof: dict[str, Any]) -> dict[str, str]:
        return scrubbed_env(self.env_allow, prof.get("secret_env", []))

    @staticmethod
    def new_session_id() -> str:
        return str(uuid.uuid4())

    # -- subprocess ---------------------------------------------------------

    async def spawn(self, argv: list[str], cwd: Path, env: dict[str, str], timeout_s: float, events_path: Path, stdin: str | None = None) -> ProcOutput:
        check_argv(argv, self.forbidden)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=self.reg.policy("harness.max_event_bytes"),
        )
        events: list[dict[str, Any]] = []
        stderr_chunks: list[bytes] = []

        async def pump_stdout() -> None:
            assert proc.stdout is not None
            with events_path.open("a", encoding="utf-8") as fh:
                async for line in proc.stdout:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    fh.write(text + "\n")
                    fh.flush()
                    try:
                        events.append(json.loads(text))
                    except json.JSONDecodeError:
                        events.append({"type": "_raw", "text": text})

        async def pump_stderr() -> None:
            assert proc.stderr is not None
            async for line in proc.stderr:
                stderr_chunks.append(line)

        async def feed() -> None:
            if stdin is not None and proc.stdin is not None:
                proc.stdin.write(stdin.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()

        timed_out = False
        try:
            await asyncio.wait_for(asyncio.gather(feed(), pump_stdout(), pump_stderr(), proc.wait()), timeout=timeout_s)
        except TimeoutError:
            timed_out = True
            _kill(proc)
            await proc.wait()
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        (events_path.parent / STDERR_FILE).write_text(stderr, encoding="utf-8")
        return ProcOutput(returncode=proc.returncode, events=events, timed_out=timed_out, stderr_tail=stderr[-2000:])

    @staticmethod
    def output_files(workspace: Path, before: set[Path]) -> list[Path]:
        return sorted(p for p in workspace.rglob("*") if p.is_file() and p not in before and p.name not in (EVENTS_FILE, STDERR_FILE))


def _kill(proc: asyncio.subprocess.Process) -> None:
    import signal

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
