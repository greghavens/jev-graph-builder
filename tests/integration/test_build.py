"""`build`: one command from documents to a verified graph; the user only approves (§15, §19).

Runs every phase with fake Jev, harnesses and embedder: S0 draft and its gate,
S1..S7, then the verification checks. P0: harnesses only generate; no harness
job makes a decision, and nothing but Jev decides.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg
import pytest

from jev_graph_builder.app import Runtime
from jev_graph_builder.build import STOPPED_AS_ASKED, Builder
from jev_graph_builder.config import CONFIG_FILE_ENV, Settings, write_config
from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.registry.loader import Registry
from tests.integration.conftest import Harnessed
from tests.integration.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

# P0: every harness job the build may run produces content; Jev decides on it.
GENERATION_PROMPTS = {"bootstrap_registry", "author_question_set", "propose_relations", "disambiguate_relations", "extract",
                      "community_summary", "repair_output", "continue_job"}


def _opener(env: Harnessed):
    @asynccontextmanager
    async def open_ctx(settings: Settings, harness: str | None) -> AsyncIterator[Context]:
        emb = Registry(settings.resolved_registry()).profile("embedding", settings.profiles.embedding)
        rt = Runtime(settings, harness, jev_transport=env.fake_jev.transport(),
                     adapters={"claude_code": env.harness, "codex": env.harness}, embedder_factory=lambda: FakeEmbedder(emb))
        try:
            yield await rt.open()
        finally:
            await rt.close()

    return open_ctx


async def test_build_end_to_end(env: Harnessed, docs: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    asked: list[tuple[str, dict[str, Any]]] = []

    def approve(kind: str, shown: dict[str, Any]) -> bool:
        asked.append((kind, shown))
        return True

    report = await Builder([str(docs[0].parent)], None, approve, "tester", open_ctx=_opener(env)).run()
    out = report.as_dict()

    kinds = [k for k, _ in asked]
    assert kinds == ["bootstrap"], kinds  # the drafted Registry is the only thing the human approves
    bootstrap = asked[0][1]
    assert bootstrap["open_decisions"] and not bootstrap["lint_errors"]
    prompts = {r.prompt_ref.split("@")[0] for r in env.harness.runs}
    assert prompts <= GENERATION_PROMPTS, f"P0: a harness job outside generation ran: {prompts - GENERATION_PROMPTS}"

    verify = out["phases"]["verify"]
    assert verify["checks"] == {"p3_violations": 0, "edge_verification_violations": 0, "pending_items": 0}, verify["checks"]
    assert out["ok"], out["stopped"]
    with psycopg.connect(env.settings.dsn) as conn:
        cost_cols = conn.execute("SELECT table_name, column_name FROM information_schema.columns "
                                 "WHERE table_schema = current_schema() AND column_name ILIKE '%%cost%%'").fetchall()
    assert cost_cols == [], f"cost is never tracked: {cost_cols}"

    # Re-running resumes: nothing is asked again and nothing new is paid for.
    runs_before = len(env.harness.runs)
    again = await Builder([str(docs[0].parent)], None, approve, "tester", open_ctx=_opener(env)).run()
    assert again.ok and len(asked) == len(kinds) and len(env.harness.runs) == runs_before


async def test_build_stage_by_stage(env: Harnessed, docs: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--stop-after: the build stops after the named stage, and re-running continues from there."""
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    src = [str(docs[0].parent)]

    first = await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env), stop_after="segment").run()
    assert first.stopped == {"reason": STOPPED_AS_ASKED, "stage": "segment"}, first.stopped
    ran = [r["stage"] for r in first.phases["graph"]["stages"]]
    assert ran == ["ingest", "segment"], ran

    jev_calls = len(env.fake_jev.calls)
    rest = await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env)).run()
    assert rest.ok, rest.stopped
    done = {r["stage"]: r for r in rest.phases["graph"]["stages"]}
    assert done["ingest"]["done"] == 0 and done["segment"]["done"] == 0, "stopped stages must not re-run"
    assert len(env.fake_jev.calls) > jev_calls


async def test_build_stops_when_not_approved(env: Harnessed, docs: list[Path], tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    active_before = Registry(env.settings.resolved_registry()).version

    report = await Builder([str(docs[0].parent)], None, lambda _k, _s: False, "tester", open_ctx=_opener(env)).run()

    assert report.stopped and report.stopped["reason"] == "not_approved" and report.stopped["gate"] == "bootstrap"
    assert "graph" not in report.phases
    assert Registry(env.settings.resolved_registry()).version == active_before  # R-080: never silently changed


@pytest.mark.parametrize(("p_related", "edges"), [(0.75, True), (0.3, False)])
async def test_build_links_follow_jev_yes_and_no(env: Harnessed, docs: list[Path], tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch, p_related: float, edges: bool) -> None:
    """If Jev says the chunks are related it is yes, if it says no it is no: nothing in between, no
    re-ask, no pause. The build runs through to verification either way."""
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    env.fake_jev.overrides.update({("link_fanout", "related"): p_related, ("link", "related"): p_related})

    report = await Builder([str(docs[0].parent)], None, lambda _k, _s: True, "tester", open_ctx=_opener(env)).run()

    assert not report.stopped, report.stopped
    assert "verify" in report.phases
    with psycopg.connect(env.settings.dsn) as conn:
        outcomes = {r[0] for r in conn.execute(
            "SELECT DISTINCT outcome FROM decisions WHERE split_part(question_set, '@', 1) IN ('link', 'link_fanout')")}
        accepted = conn.execute("SELECT count(*) FROM edges WHERE NOT structural AND status = 'accepted'").fetchone()[0]
    assert outcomes <= {"accept", "reject"}
    assert (accepted > 0) == edges
    assert env.fake_jev.calls and all(r.prompt_ref.split("@")[0] in GENERATION_PROMPTS for r in env.harness.runs)


@pytest.mark.parametrize(("p_applies", "derived"), [(0.97, ["1.0", "2.0"]), (0.3, None)])
async def test_build_derives_missing_metadata_from_text(env: Harnessed, docs: list[Path], tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch, p_applies: float,
                                                        derived: list[str] | None) -> None:
    """§8.1: a document without a `metadata_fields` key gets the corpus values its text names word for
    word (code, whatever Jev would say); otherwise the values Jev says its text applies to. The
    candidates come from the other documents, never from code. Jev says no → left untagged."""
    import yaml

    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    pol_path = env.settings.resolved_registry() / "policies.yaml"
    pol = yaml.safe_load(pol_path.read_text(encoding="utf-8"))
    pol["ingest"]["metadata_fields"] = ["release"]
    pol_path.write_text(yaml.safe_dump(pol, sort_keys=False), encoding="utf-8")
    tagged = {docs[0].name: "1.0", docs[1].name: "2.0"}
    for d in docs[:2]:
        d.write_text(f'---\nrelease: "{tagged[d.name]}"\n---\n' + d.read_text(encoding="utf-8"), encoding="utf-8")
    named = docs[2].name
    docs[2].write_text(docs[2].read_text(encoding="utf-8") + "\n\nThis also holds for release 2.0.\n", encoding="utf-8")
    env.fake_jev.overrides[("metadata_applies", "applies")] = p_applies

    report = await Builder([str(docs[0].parent)], None, lambda _k, _s: True, "tester", open_ctx=_opener(env),
                           stop_after="segment").run()
    assert report.stopped == {"reason": STOPPED_AS_ASKED, "stage": "segment"}, report.stopped

    with psycopg.connect(env.settings.dsn) as conn:
        rows = conn.execute("SELECT d.source_uri, c.meta FROM chunks c JOIN documents d USING (doc_id)").fetchall()
        asked = conn.execute("SELECT count(*) FROM decisions "
                             "WHERE split_part(question_set, '@', 1) = 'metadata_applies'").fetchone()
    assert rows
    for uri, meta in rows:
        name = Path(uri).name
        if name in tagged:
            assert meta == {"release": tagged[name]}, (name, meta)
        elif name == named:
            assert meta == {"release": ["2.0"]}, (name, meta)
        else:
            assert (meta or {}).get("release") == derived, (name, meta)
    assert asked[0] > 0  # only Jev decided


async def test_build_segments_by_code_without_asking_jev(env: Harnessed, docs: list[Path], tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """§8.3: code cuts where a section starts, and splits every section longer than a 16-token cap
    (all fixture sections are) at unit boundaries. Jev is asked about no gap; each chunk gets one chunk
    check. Every chunk of two or more units fits the cap, and the chunks cover each document's units
    in order with nothing lost."""
    import yaml

    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    pol_path = env.settings.resolved_registry() / "policies.yaml"
    pol = yaml.safe_load(pol_path.read_text(encoding="utf-8"))
    pol["segment"]["max_fraction"] = 0.0625                  # a 16-token cap: below every fixture section
    pol_path.write_text(yaml.safe_dump(pol, sort_keys=False), encoding="utf-8")
    reg = Registry(env.settings.resolved_registry())
    cap = int(reg.profile("embedding", env.settings.profiles.embedding)["max_tokens"] * pol["segment"]["max_fraction"])

    report = await Builder([str(docs[0].parent)], None, lambda _k, _s: True, "tester", open_ctx=_opener(env),
                           stop_after="segment").run()
    assert report.stopped == {"reason": STOPPED_AS_ASKED, "stage": "segment"}, report.stopped

    with psycopg.connect(env.settings.dsn) as conn:
        chunks = conn.execute("SELECT doc_id, ord, unit_ids, tokens FROM chunks ORDER BY doc_id, ord").fetchall()
        units = {r[0]: r[1] for r in conn.execute(
            "SELECT doc_id, array_agg(unit_id ORDER BY ord) FROM units GROUP BY doc_id").fetchall()}
        by_qs = dict(conn.execute("SELECT split_part(question_set, '@', 1), count(*) FROM decisions GROUP BY 1").fetchall())
    by_doc: dict[str, list[str]] = {}
    for doc_id, _ord, unit_ids, tokens in chunks:
        assert len(unit_ids) == 1 or tokens <= cap, (doc_id, tokens, cap)
        by_doc.setdefault(doc_id, []).extend(unit_ids)
    assert by_doc == units                                    # every unit, once, in order
    assert len(chunks) > len(units)                           # code split the long sections
    assert not {"segment", "segment_fanout"} & set(by_qs)     # Jev decided no gap
    assert by_qs["chunk_check_fanout"] == len(chunks)         # one per chunk


async def test_a_changed_file_supersedes_the_old_versions_chunks(env: Harnessed, docs: list[Path], tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """§14.3: a changed file is a new document; the old version, its chunks and what was drawn from them
    (edges, mentions, claims) are superseded, so no later stage selects them."""
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    src = [str(docs[0].parent)]
    like = f"%{docs[0].name}"

    first = await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env)).run()
    assert first.ok, first.stopped
    with psycopg.connect(env.settings.dsn) as conn:
        (old,) = conn.execute("SELECT doc_id FROM documents WHERE source_uri LIKE %s", (like,)).fetchone()
        drawn = conn.execute("SELECT (SELECT count(*) FROM mentions m JOIN chunks c USING (chunk_id) WHERE c.doc_id = %s), "
                             "(SELECT count(*) FROM claims x JOIN chunks c USING (chunk_id) WHERE c.doc_id = %s)",
                             (old, old)).fetchone()
    assert all(drawn), drawn  # the old version has mentions and claims to supersede

    docs[0].write_text(docs[0].read_text(encoding="utf-8") + "\n\nAn added closing paragraph.\n", encoding="utf-8")
    second = await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env), stop_after="segment").run()
    assert second.stopped == {"reason": STOPPED_AS_ASKED, "stage": "segment"}, second.stopped

    def live(conn: psycopg.Connection, sql: str) -> int:
        return conn.execute(sql + " AND x.status <> 'superseded'", (old,)).fetchone()[0]

    with psycopg.connect(env.settings.dsn) as conn:
        assert conn.execute("SELECT status FROM documents WHERE doc_id = %s", (old,)).fetchone()[0] == "superseded"
        assert live(conn, "SELECT count(*) FROM chunks x WHERE x.doc_id = %s") == 0
        assert live(conn, "SELECT count(*) FROM edges x JOIN chunks c ON c.chunk_id IN (x.src_id, x.dst_id) WHERE c.doc_id = %s") == 0
        assert live(conn, "SELECT count(*) FROM mentions x JOIN chunks c USING (chunk_id) WHERE c.doc_id = %s") == 0
        assert live(conn, "SELECT count(*) FROM claims x JOIN chunks c USING (chunk_id) WHERE c.doc_id = %s") == 0
        assert live(conn, "SELECT count(*) FROM entities x WHERE NOT EXISTS (SELECT 1 FROM mentions m WHERE "
                          "m.entity_id = x.entity_id AND m.status <> 'superseded') AND EXISTS (SELECT 1 FROM mentions m "
                          "JOIN chunks c USING (chunk_id) WHERE m.entity_id = x.entity_id AND c.doc_id = %s)") == 0
        new_chunks = conn.execute("SELECT count(*) FROM chunks c JOIN documents d USING (doc_id) WHERE d.source_uri LIKE %s "
                                  "AND d.doc_id <> %s AND c.status = 'accepted'", (like, old)).fetchone()[0]
    assert new_chunks > 0


class Killed(Exception):
    pass


async def test_build_evolves_relations_jev_keeps_and_survives_a_kill(env: Harnessed, docs: list[Path], tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """§8.7.4: Jev picks `other` for every link; the harness proposes a relation citing those pairs; Jev
    confirms it on each, so it is kept and activated as Jev's decision. A kill after Jev's approval is
    committed but before activation loses nothing: the next build activates it."""
    from jev_graph_builder.registry import gate

    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    env.fake_jev.overrides.update({("link", "relation"): "other", ("link_fanout", "relation"): "other"})
    env.harness.relation_proposal = {
        "name": "fixture_link", "definition": "The target continues an account the source begins.",
        "examples": ["A runbook step and the incident it resolves."], "direction": "directed",
        "direction_semantics": "source begins, target continues"}
    activate = gate.activate

    async def killed_on_jev_activation(ctx: Context, proposal: Any) -> dict[str, Any]:
        row = await ctx.db.fetchone("SELECT approved_by FROM registry_versions WHERE version = %s", (proposal.version,))
        if row and row["approved_by"] == "jev":
            raise Killed
        return await activate(ctx, proposal)

    monkeypatch.setattr(gate, "activate", killed_on_jev_activation)
    src = [str(docs[0].parent)]
    with pytest.raises(Killed):
        await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env)).run()

    with psycopg.connect(env.settings.dsn) as conn:
        pending = conn.execute("SELECT version FROM registry_versions WHERE approved_by = 'jev' AND status = 'approved'").fetchall()
        seen = conn.execute("SELECT count(*) FROM edges WHERE rel = 'other' AND evolution_seen").fetchone()[0]
    assert len(pending) == 1 and seen > 0  # Jev's approval and the examined picks committed together

    monkeypatch.setattr(gate, "activate", activate)
    report = await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env)).run()

    assert report.ok, report.stopped
    active = Registry(env.settings.resolved_registry())
    assert "fixture_link" in {r["name"] for r in active.ontology("relation_types")}
    with psycopg.connect(env.settings.dsn) as conn:
        status = conn.execute("SELECT status FROM registry_versions WHERE version = %s", (pending[0][0],)).fetchone()[0]
    assert status == "active"


async def test_enrich_writes_each_batch_as_its_extraction_finishes(env: Harnessed, docs: list[Path], tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk is verified and written as soon as its own extraction job finishes, while other jobs
    are still running, so a stopped run keeps every chunk already written."""
    import asyncio

    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    src = [str(docs[0].parent)]
    assert (await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env), stop_after="segment").run()).stopped

    policy = Registry.policy
    monkeypatch.setattr(Registry, "policy", lambda self, key: 1 if key == "extract.batch_chunks" else policy(self, key))
    run = env.harness.run
    extract_jobs: list[bool] = []

    async def later_jobs_wait_for_a_written_chunk(spec: Any) -> Any:
        if env.harness.reg.prompt(spec.prompt_ref).name == "extract":
            first = not extract_jobs
            extract_jobs.append(first)
            if not first:
                async with await psycopg.AsyncConnection.connect(env.settings.dsn) as conn:
                    for _ in range(200):
                        cur = await conn.execute("SELECT count(*) FROM work_items WHERE stage = 'enrich' AND status = 'done'")
                        if (await cur.fetchone())[0]:
                            break
                        await asyncio.sleep(0.05)
                    else:
                        raise AssertionError("no enriched chunk was written while later extraction jobs ran")
        return await run(spec)

    monkeypatch.setattr(env.harness, "run", later_jobs_wait_for_a_written_chunk)
    report = await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env), stop_after="enrich").run()

    enrich = {r["stage"]: r for r in report.phases["graph"]["stages"]}["enrich"]
    assert len(extract_jobs) > 1 and enrich["failed"] == 0 and enrich["done"] == len(extract_jobs), enrich


async def test_enrich_fails_chunks_without_extraction_and_stops_on_a_usage_limit(
        env: Harnessed, docs: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk whose extraction produced nothing fails (a re-run retries it) instead of being written
    empty; the harness usage limit stops the run, and no further extraction job starts."""
    from jev_graph_builder.harness.base import HarnessUsageLimit

    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    src = [str(docs[0].parent)]
    assert (await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env), stop_after="segment").run()).stopped

    policy = Registry.policy
    one = {"extract.batch_chunks", "run.concurrency.harness_jobs"}
    monkeypatch.setattr(Registry, "policy", lambda self, key: 1 if key in one else policy(self, key))
    run = env.harness.run
    jobs: list[Path] = []

    async def empty_then_limited(spec: Any) -> Any:
        if env.harness.reg.prompt(spec.prompt_ref).name != "extract":
            return await run(spec)
        if spec.workspace not in jobs:
            jobs.append(spec.workspace)
        if spec.workspace != jobs[0]:
            raise HarnessUsageLimit("five_hour usage limit, resets at 1790268600")
        result = await run(spec)
        (spec.workspace / spec.variables["output_file"]).unlink()
        result.ok, result.error = False, "no output"
        return result

    monkeypatch.setattr(env.harness, "run", empty_then_limited)
    report = await Builder(src, None, lambda _k, _s: True, "tester", open_ctx=_opener(env), stop_after="enrich").run()

    assert report.stopped and "HarnessUsageLimit" in str(report.stopped), report.stopped
    assert len(jobs) == 2  # the empty job, then the limited one; nothing after the limit
    async with await psycopg.AsyncConnection.connect(env.settings.dsn) as conn:
        cur = await conn.execute("SELECT status, last_error FROM work_items WHERE stage = 'enrich'")
        rows = await cur.fetchall()
        cur = await conn.execute("SELECT count(*) FROM chunks WHERE summary_decision_ids IS NOT NULL")
        written = (await cur.fetchone())[0]
    assert rows and not any(status == "done" for status, _ in rows), rows
    assert sum(1 for _, err in rows if err and "ExtractionMissing" in err) == 1, rows
    assert written == 0
