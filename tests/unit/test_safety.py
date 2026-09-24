"""R-002 hard-coding lint, R-020 environment scrub, R-021 forbidden harness flags."""
from __future__ import annotations

import pytest

from jev_graph_builder.harness.base import ForbiddenArgument, check_argv, scrubbed_env
from jev_graph_builder.lint.hardcoding import DEFAULT_ALLOWLIST, Allowlist, lint_package, lint_source
from tests.conftest import ROOT


def allow() -> Allowlist:
    return Allowlist.load(ROOT / DEFAULT_ALLOWLIST)


def test_package_has_no_hardcoding():
    assert lint_package(ROOT / "jev_graph_builder", allow()) == []


def test_lint_flags_domain_strings_and_thresholds():
    src = 'def f(p):\n    label = "the gateway service handles retries"\n    return p >= 0.87\n'
    v = lint_source(src, "pipeline/x.py", allow())
    kinds = " ".join(str(x) for x in v)
    assert len(v) == 2 and "0.87" in kinds


def test_env_scrub_drops_session_and_unlisted(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION", "s")
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-value")
    monkeypatch.setenv("HOME", "/home/x")
    env = scrubbed_env(["HOME", "CLAUDE_CODE_SESSION"], [])
    assert env == {"HOME": "/home/x"}


@pytest.mark.parametrize("arg", ["--dangerously-skip-permissions", "--yolo", "sandbox_mode=danger-full-access", "bypassPermissions"])
def test_forbidden_args_refused(reg, arg):
    with pytest.raises(ForbiddenArgument):
        check_argv(["claude", "-p", arg], reg.policy("harness.forbidden_args"))


def test_spawn_reads_event_lines_beyond_asyncio_default(reg, tmp_path):
    """A harness's final result event carries the whole structured output on one line."""
    import asyncio
    import sys

    from jev_graph_builder.harness.claude import ClaudeCodeAdapter

    big = "x" * (1 << 20)
    script = "import json; print(json.dumps({'type': 'result', 'text': 'x' * (1 << 20)}))"
    out = asyncio.run(ClaudeCodeAdapter(reg).spawn(
        [sys.executable, "-c", script], cwd=tmp_path, env={}, timeout_s=60, events_path=tmp_path / "events.jsonl"))
    assert out.returncode == 0 and out.events[0]["text"] == big
