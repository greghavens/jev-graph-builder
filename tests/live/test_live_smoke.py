"""Live smoke tests (§19): real Jev and real harness CLIs, behind JEV_GRAPH_BUILDER_LIVE=1.

They call the real services and need `TYPESAFE_API_KEY` plus logged-in `claude` and `codex`
binaries, so they never run by default. The parity test is acceptance item 8: the
same extraction job must pass schema validation on both harnesses.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jev_graph_builder.harness.claude import ClaudeCodeAdapter
from jev_graph_builder.harness.codex import CodexAdapter
from jev_graph_builder.harness.jobs import JobRunner
from jev_graph_builder.jev.client import build_provider
from jev_graph_builder.jev.service import JevService
from jev_graph_builder.parse.parsers import parse_markdown
from jev_graph_builder.pipeline.s3_enrich import JOB, PROMPT
from jev_graph_builder.registry.loader import Registry

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests/fixtures/corpus"

pytestmark = pytest.mark.skipif(os.environ.get("JEV_GRAPH_BUILDER_LIVE") != "1", reason="live smoke tests need JEV_GRAPH_BUILDER_LIVE=1")


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry(ROOT / "registry")


def _passages(n: int) -> list[str]:
    texts = []
    for path in sorted(CORPUS.iterdir())[:n]:
        units = parse_markdown(path.read_text(encoding="utf-8")).units
        texts.append("\n\n".join(u.text for u in units[:4]))
    return texts


async def test_live_jev_call_parses_and_pins_model(reg: Registry) -> None:
    profile = dict(reg.profile("jev", "typesafe"))
    provider = build_provider(profile, reg.policies)
    raw = await provider.call({"passage": "The sky is blue."}, {"q1": {"type": "noul", "instructions": "`passage` names a colour."}})
    assert raw.model == profile["model"]  # R-010: the pinned versioned ID answers, not an alias
    jev = JevService(reg, provider, db=None)
    passage = _passages(1)[0]
    result = await jev.ask("rag_passage", {"query": "What is ledgerd?", "passage": passage}, "smoke", "live-1")
    d = result.single
    qs = result.qs
    assert set(d.answers) == set(qs.questions)
    for name, ans in d.answers.items():
        assert ans["type"] == qs.questions[name]["type"]
        assert 0.0 <= ans["p"] <= 1.0


@pytest.mark.parametrize("profile", ["cc_batch", "codex_batch"])
async def test_live_harness_parity_extraction(reg: Registry, tmp_path: Path, profile: str) -> None:
    runner = JobRunner(reg, {"claude_code": ClaudeCodeAdapter(reg), "codex": CodexAdapter(reg)}, db=None, workspace_root=tmp_path)
    records = [{"id": f"live-{i}", "text": t, "heading_path": []} for i, t in enumerate(_passages(2))]
    ontology_file = reg.policy("extract.ontology_file")
    files = {ontology_file: {"entity_types": reg.ontology("entity_types"), "claim_types": reg.ontology("claim_types")}}
    outcome = await runner.run_job(JOB, PROMPT, records, profile, variables={"ontology_file": ontology_file}, files=files)
    assert not outcome.missing, outcome.missing
    assert set(outcome.records) == {r["id"] for r in records}
