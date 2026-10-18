"""The Cloudflare Worker's TypeScript gives what the Python query path gives: the same state for the same inputs
(fields picked, text truncated, untrusted text wrapped), the same decision IDs, the same gate outcomes and the
same score levels. Python writes the expected values; the Worker's own test (`npm test`) checks them."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from jev_graph_builder.cloudflare import bundle
from jev_graph_builder.cloudflare.publish import WORKER_SRC
from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.jev.gating import evaluate, top_level
from jev_graph_builder.pipeline.common import Context

pytestmark = pytest.mark.integration

LONG = "VCF 9.0 upgrade 🚀 – NSX Manager certificate rotation. " * 400  # over every state token limit, with non-BMP text


def _cases(ctx: Context, b: dict) -> list[dict]:
    scope = ctx.reg.corpus["scope"]
    inputs = {
        "query_route": {"query": LONG, "corpus": {"scope": scope}},
        "rag_passage": {"query": "How do I rotate the NSX certificate?", "passage": LONG, "other_evidence": [LONG, "short peer"]},
        "rag_passage.no_peers": {"query": "How do I rotate the NSX certificate?", "passage": LONG, "other_evidence": None},
        "expand_gate": {"query": "rotate", "from_chunk": {"title": "Certificates", "text": LONG},
                        "relation": {"name": "elaborates", "definition": "adds detail"}, "neighbor": {"title": "", "text": LONG}},
    }
    out = []
    for name, inp in inputs.items():
        qs = ctx.reg.question_set(name.split(".")[0])
        state = ctx.jev.states.build(qs, inp)
        want = b["question_sets"][name]
        out.append({"qs": name, "inputs": inp, "state": state,
                    "decision_id": sha256_hex("query", "subject", qs.ref, sha256_hex(state), want["questions_hash"], "")})
    return out


def _gates(ctx: Context) -> list[dict]:
    bar = ctx.jev.bar(0)
    out = []
    for name, q in (("rag_passage", "relevant"), ("expand_gate", "helps")):
        qs = ctx.reg.question_set(name)
        for p in (0.2, 0.5, 0.500001, 0.9):
            answers = {q: {"type": "noul", "p": p}}
            out.append({"qs": name, "answers": answers, "outcome": evaluate(qs, answers, bar).outcome})
    return out


async def test_worker_matches_the_python_query_path(ctx: Context, tmp_path: Path) -> None:
    if not (WORKER_SRC / "node_modules").exists() or shutil.which("npm") is None:
        pytest.skip("the Worker's npm dependencies are not installed (npm install in jev_graph_builder/cloudflare/worker)")
    b = bundle.build(ctx, ctx.reg.profile("embedding", ctx.reg.policy("publish.cloudflare.serving_embedding")))
    levels = [{"probabilities": {"1": 0.2, "2": 0.5, "3": 0.3}}, {"probabilities": {"1": 0.4, "2": 0.4, "3": 0.2}}, {"score": 2.6}]
    fixtures = {"bundle": b, "states": _cases(ctx, b), "gates": _gates(ctx),
                "levels": [{"answer": a, "level": top_level({**a, "probabilities": {int(k): v for k, v in a.get("probabilities", {}).items()}})}
                           for a in levels]}
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(fixtures, ensure_ascii=False))
    done = subprocess.run(["npm", "test", "--silent"], cwd=WORKER_SRC, capture_output=True, text=True,
                          env={**os.environ, "JGB_PARITY_FIXTURES": str(path)})
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
