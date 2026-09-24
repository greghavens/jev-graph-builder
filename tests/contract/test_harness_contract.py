"""Contract tests: harness adapters against fake CLIs that replay recorded event streams (§20).

Each fake binary records its argv, stdin and environment, then prints the
events of the documented stream format (`claude -p --output-format stream-json`,
`codex exec --json`).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from jev_graph_builder.harness.base import HarnessError, RunSpec
from jev_graph_builder.harness.claude import ClaudeCodeAdapter
from jev_graph_builder.harness.codex import CodexAdapter
from jev_graph_builder.registry.loader import Registry

ROOT = Path(__file__).resolve().parents[2]
ANSWER = {"answer_sentences": [{"text": "Version 2.4 adds HNSW indexes.", "citation_ids": ["c1"]}]}

CLAUDE_EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "SID", "model": "claude-sonnet-5"},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}, "session_id": "SID"},
    {"type": "result", "subtype": "success", "is_error": False, "session_id": "SID", "result": "done",
     "structured_output": ANSWER, "num_turns": 2,
     "usage": {"input_tokens": 1200, "output_tokens": 80}},
]

CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "th_123"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": json.dumps(ANSWER)}},
    {"type": "turn.completed", "usage": {"input_tokens": 900, "cached_input_tokens": 100, "output_tokens": 60}},
]


def _fake_cli(tmp: Path, name: str, events: list[dict]) -> Path:
    script = tmp / name
    record = tmp / f"{name}.calls.jsonl"
    body = f"""#!{sys.executable}
import json, os, sys
events = {json.dumps(events)!r}
argv = sys.argv[1:]
stdin = sys.stdin.read()
with open({str(record)!r}, "a") as fh:
    fh.write(json.dumps({{"argv": argv, "stdin": stdin, "env": sorted(os.environ)}}) + "\\n")
out = json.loads(events)
sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv else None
for ev in out:
    if sid and "session_id" in ev:
        ev["session_id"] = sid
    print(json.dumps(ev), flush=True)
"""
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _calls(script: Path) -> list[dict]:
    return [json.loads(line) for line in Path(f"{script}.calls.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry(ROOT / "registry")


def _spec(tmp: Path, profile: str) -> RunSpec:
    return RunSpec(
        prompt_ref="answer",
        variables={"query": "What does 2.4 add?", "passages": [{"id": "c1", "text": "Version 2.4 adds HNSW indexes."}], "failed_sentences": []},
        output_schema_ref="answer",
        workspace=tmp / "ws",
        profile=profile,
    )


async def test_claude_run_structured_output_and_safe_invocation(tmp_path: Path, reg: Registry, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "must-not-leak")
    fake = _fake_cli(tmp_path, "claude", CLAUDE_EVENTS)
    adapter = ClaudeCodeAdapter(reg, binary=str(fake))
    result = await adapter.run(_spec(tmp_path, "cc_readonly"))

    assert result.ok, result.error
    assert result.structured == ANSWER
    assert result.events_path.is_file() and len(result.events_path.read_text().splitlines()) == len(CLAUDE_EVENTS)
    call = _calls(fake)[0]
    argv = call["argv"]
    assert "-p" in argv and argv[argv.index("--output-format") + 1] == "stream-json"
    sent = json.loads(argv[argv.index("--json-schema") + 1])
    assert "$schema" not in sent  # the CLI rejects the draft meta-schema URI
    assert sent == {k: v for k, v in reg.schema("answer").items() if k != "$schema"}
    assert argv[argv.index("--permission-mode") + 1] == "plan"
    assert "What does 2.4 add?" in call["stdin"] and "What does 2.4 add?" not in " ".join(argv)
    assert not any(v.startswith(("CLAUDE", "CODEX")) for v in call["env"])
    assert "TYPESAFE_API_KEY" not in call["env"]


async def test_claude_resume_passes_the_session(tmp_path: Path, reg: Registry) -> None:
    first = _fake_cli(tmp_path, "claude", CLAUDE_EVENTS)
    adapter = ClaudeCodeAdapter(reg, binary=str(first))
    spec = _spec(tmp_path, "cc_readonly")
    r1 = await adapter.run(spec)
    second = _fake_cli(tmp_path, "claude2", CLAUDE_EVENTS)
    adapter2 = ClaudeCodeAdapter(reg, binary=str(second))
    r2 = await adapter2.resume(r1.session_id, spec)
    assert r2.ok, r2.error
    argv = _calls(second)[0]["argv"]
    assert argv[argv.index("--resume") + 1] == r1.session_id


async def test_claude_success_without_structured_output_fails(tmp_path: Path, reg: Registry) -> None:
    events = json.loads(json.dumps(CLAUDE_EVENTS))
    del events[-1]["structured_output"]
    adapter = ClaudeCodeAdapter(reg, binary=str(_fake_cli(tmp_path, "claude", events)))
    result = await adapter.run(_spec(tmp_path, "cc_readonly"))
    assert not result.ok and "structured_output" in result.error


async def test_forbidden_bypass_flags_are_refused(tmp_path: Path, reg: Registry) -> None:
    fake = _fake_cli(tmp_path, "claude", CLAUDE_EVENTS)
    adapter = ClaudeCodeAdapter(reg, binary=str(fake))
    bad = {**reg.profile("harness", "cc_readonly"), "extra_args": ["--dangerously-skip-permissions"]}
    adapter.profile = lambda name: bad  # type: ignore[method-assign]
    with pytest.raises(HarnessError):
        await adapter.run(_spec(tmp_path, "cc_readonly"))
    assert not Path(f"{fake}.calls.jsonl").exists()


async def test_codex_exec_json_output_schema(tmp_path: Path, reg: Registry) -> None:
    fake = _fake_cli(tmp_path, "codex", CODEX_EVENTS)
    adapter = CodexAdapter(reg, binary=str(fake))
    result = await adapter.run(_spec(tmp_path, "codex_readonly"))
    assert result.ok, result.error
    assert result.structured == ANSWER
    assert result.session_id == "th_123"
    assert result.usage["input_tokens"] == 900
    argv = _calls(fake)[0]["argv"]
    assert argv[:2] == ["exec", "--json"]
    schema_file = Path(argv[argv.index("--output-schema") + 1])
    assert json.loads(schema_file.read_text()) == reg.schema("answer")
    assert argv[argv.index("-s") + 1] == reg.profile("harness", "codex_readonly")["sandbox"]
    assert argv[-1] == "-"  # prompt on stdin


async def test_codex_invalid_output_gets_one_repair_turn(tmp_path: Path, reg: Registry) -> None:
    events = json.loads(json.dumps(CODEX_EVENTS))
    events[2]["item"]["text"] = json.dumps({"answer_sentences": [{"text": "no citations"}]})
    fake = _fake_cli(tmp_path, "codex", events)
    result = await CodexAdapter(reg, binary=str(fake)).run(_spec(tmp_path, "codex_readonly"))
    calls = _calls(fake)
    assert len(calls) == 2
    assert calls[1]["argv"][:2] == ["exec", "resume"] and "th_123" in calls[1]["argv"]
    assert not result.ok and result.extra["repaired_from_error"].startswith("schema")


async def test_codex_relative_workspace_is_passed_absolute(tmp_path: Path, reg: Registry, monkeypatch) -> None:
    """The CLI runs with cwd=workspace, so a relative `-C` would resolve against itself."""
    monkeypatch.chdir(tmp_path)
    fake = _fake_cli(tmp_path, "codex", CODEX_EVENTS)
    spec = _spec(Path("rel"), "codex_readonly")
    result = await CodexAdapter(reg, binary=str(fake)).run(spec)
    assert result.ok, result.error
    argv = _calls(fake)[0]["argv"]
    for flag in ("-C", "-o", "--output-schema"):
        assert Path(argv[argv.index(flag) + 1]).is_absolute(), flag
    assert Path(argv[argv.index("-C") + 1]) == tmp_path / "rel" / "ws"


async def test_codex_failure_before_thread_start_has_no_session(tmp_path: Path, reg: Registry) -> None:
    """With no `thread.started`, there is no Codex session to resume or repair."""
    fake = _fake_cli(tmp_path, "codex", [])
    result = await CodexAdapter(reg, binary=str(fake)).run(_spec(tmp_path, "codex_readonly"))
    assert not result.ok and result.session_id is None
    assert len(_calls(fake)) == 1  # no repair turn against a session that never existed

def test_fake_cli_is_executable(tmp_path: Path) -> None:
    assert os.access(_fake_cli(tmp_path, "x", []), os.X_OK)
