"""`publish`: the built graph replaces the hosted server's database without touching it in place (§9)."""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from jev_graph_builder.app import migration_params
from jev_graph_builder.build import Builder
from jev_graph_builder.config import CONFIG_FILE_ENV, write_config
from jev_graph_builder.mcp_auth import _SCHEMA as OAUTH_SCHEMA
from jev_graph_builder.store.publish import publish
from tests.integration.conftest import Harnessed
from tests.integration.test_build import _opener

pytestmark = pytest.mark.integration


async def test_publish_swaps_in_the_graph_and_keeps_sign_ins_and_the_previous_copy(
        env: Harnessed, docs: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_dsn: str) -> None:
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    env.fake_jev.overrides.update({("link_fanout", "related"): 0.75, ("link", "related"): 0.75})
    built = await Builder([str(docs[0].parent)], None, lambda _k, _s: True, "tester", open_ctx=_opener(env)).run()
    assert built.ok, built.stopped

    name = "jgb_pub_" + uuid.uuid4().hex[:8]
    to = make_conninfo(**{**conninfo_to_dict(server_dsn), "dbname": name})
    with psycopg.connect(env.settings.dsn) as conn:
        # A rejected edge, which publish leaves out.
        conn.execute("UPDATE edges SET status = 'rejected' WHERE edge_id = (SELECT min(edge_id) FROM edges WHERE NOT structural)")
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        accepted = conn.execute("SELECT count(*) FROM edges WHERE status = 'accepted'").fetchone()[0]
    try:
        async with _opener(env)(env.settings, None) as ctx:
            first = await publish(ctx.db, migration_params(ctx.reg, ctx.settings), to)
        assert first["previous"] is None and first["rows"]["chunks"] == chunks and first["rows"]["edges"] == accepted
        with psycopg.connect(to, autocommit=True) as conn:
            conn.execute(OAUTH_SCHEMA)
            conn.execute("INSERT INTO mcp_oauth (kind, key, data) VALUES ('client', 'c1', '{}')")
            conn.execute("CREATE TABLE marker (x int)")  # only in the first copy

        async with _opener(env)(env.settings, None) as ctx:
            second = await publish(ctx.db, migration_params(ctx.reg, ctx.settings), to)
        assert second["previous"] == f"{name}_prev" and second["mcp_sign_ins_carried"] == 1
        with psycopg.connect(to) as conn:
            assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == chunks
            assert conn.execute("SELECT count(*) FROM edges WHERE status <> 'accepted'").fetchone()[0] == 0
            assert conn.execute("SELECT key FROM mcp_oauth").fetchall() == [("c1",)]
            assert conn.execute("SELECT to_regclass('marker')").fetchone()[0] is None, "the database was written in place"
            assert conn.execute("SELECT count(*) FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'").fetchone()[0] == 1
        with psycopg.connect(make_conninfo(to, dbname=f"{name}_prev")) as conn:
            assert conn.execute("SELECT to_regclass('marker')").fetchone()[0] is not None
    finally:
        with psycopg.connect(server_dsn, autocommit=True) as conn:
            for db in (name, f"{name}_prev", f"{name}_next"):
                conn.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')


async def test_an_interrupted_publish_resumes_without_reloading_finished_tables(
        env: Harnessed, docs: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_dsn: str) -> None:
    from jev_graph_builder.store import publish as pub

    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "jev-graph-builder.yaml"))
    write_config(env.settings.model_dump(mode="json"))
    env.fake_jev.overrides.update({("link_fanout", "related"): 0.75, ("link", "related"): 0.75})
    built = await Builder([str(docs[0].parent)], None, lambda _k, _s: True, "tester", open_ctx=_opener(env)).run()
    assert built.ok, built.stopped
    name = "jgb_pub_" + uuid.uuid4().hex[:8]
    to = make_conninfo(**{**conninfo_to_dict(server_dsn), "dbname": name})
    real_copy, copied = pub._copy, []

    async def killed_at_chunks(source, dst, table, where=""):
        if table == "chunks":
            raise RuntimeError("killed")
        copied.append(table)
        return await real_copy(source, dst, table, where)

    try:
        monkeypatch.setattr(pub, "_copy", killed_at_chunks)
        async with _opener(env)(env.settings, None) as ctx:
            with pytest.raises(RuntimeError, match="killed"):
                await publish(ctx.db, migration_params(ctx.reg, ctx.settings), to)
        finished = list(copied)
        assert finished and "chunks" not in finished

        copied.clear()
        monkeypatch.setattr(pub, "_copy", lambda *a, **k: (copied.append(a[2]), real_copy(*a, **k))[1])
        async with _opener(env)(env.settings, None) as ctx:
            done = await publish(ctx.db, migration_params(ctx.reg, ctx.settings), to)
        assert not set(copied) & set(finished), "a finished table was loaded again"
        assert "chunks" in copied
        with psycopg.connect(env.settings.dsn) as conn:
            assert done["rows"]["chunks"] == conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        with psycopg.connect(to) as conn:
            assert conn.execute("SELECT count(*) FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'").fetchone()[0] == 1
    finally:
        with psycopg.connect(server_dsn, autocommit=True) as conn:
            for db in (name, f"{name}_prev", f"{name}_next"):
                conn.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
