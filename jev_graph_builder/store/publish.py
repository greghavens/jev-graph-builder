"""Publish the graph to the database a hosted query server reads (`jev-graph-builder publish`).

The target database `<name>` is never written in place. The graph is loaded into `<name>_next` on the
same server, the MCP sign-ins in `<name>` are carried over, then `<name>_next` takes its place in two
renames; the database it replaces stays as `<name>_prev`. Only what the query path reads is copied,
and of the edges only the accepted ones.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import tuple_row

from jev_graph_builder.mcp_auth import _SCHEMA as OAUTH_SCHEMA
from jev_graph_builder.store import migrate
from jev_graph_builder.store.db import Database

# Tables the query path reads, copied whole; edges are copied separately (accepted rows only).
TABLES = ("registry_versions", "corpora", "documents", "chunks", "units", "communities")
EDGES = "edges"
OAUTH_TABLE = "mcp_oauth"
# In `<name>_next`: the tables loaded so far and from how many source rows, so a rerun resumes.
PROGRESS_TABLE = "jgb_publish_progress"


class PublishError(Exception):
    pass


def _dsn(conninfo: dict[str, Any], dbname: str) -> str:
    return make_conninfo(**{**conninfo, "dbname": dbname})


async def _exists(admin: psycopg.AsyncConnection, name: str) -> bool:
    cur = await admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
    return await cur.fetchone() is not None


async def _columns(conn: psycopg.AsyncConnection, table: str) -> list[str]:
    cur = await conn.cursor(row_factory=tuple_row).execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = %s "
        "AND is_generated = 'NEVER' "  # generated columns are computed on the target
        "ORDER BY ordinal_position", (table,))
    return [r[0] for r in await cur.fetchall()]


async def _copy(source: Database, dst: psycopg.AsyncConnection, table: str, where: str = "") -> int:
    """Stream one table from the build database into the (empty) table in `dst` (binary COPY: both sides run the
    same migrations). Loaded and recorded in PROGRESS_TABLE in one transaction, so it is either done or not."""
    async with source.tx() as src:
        cols = await _columns(src, table)
        missing = set(cols) - set(await _columns(dst, table))
        if missing:
            raise PublishError(f"{table}: the target schema has no column(s) {sorted(missing)}")
        names = sql.SQL(", ").join(map(sql.Identifier, cols))
        out = sql.SQL("COPY (SELECT {} FROM {} {}) TO STDOUT (FORMAT BINARY)").format(names, sql.Identifier(table), sql.SQL(where))
        into = sql.SQL("COPY {} ({}) FROM STDIN (FORMAT BINARY)").format(sql.Identifier(table), names)
        async with dst.transaction():
            async with src.cursor().copy(out) as reader, dst.cursor().copy(into) as writer:
                async for block in reader:
                    await writer.write(block)
            cur = await dst.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
            rows = (await cur.fetchone())[0]
            await dst.execute(f"INSERT INTO {PROGRESS_TABLE} (table_name, source_rows) VALUES (%s, %s) "
                              "ON CONFLICT (table_name) DO UPDATE SET source_rows = EXCLUDED.source_rows", (table, rows))
    return rows


async def _source_rows(source: Database, table: str, where: str) -> int:
    row = await source.fetchone(f"SELECT count(*) AS n FROM {table} {where}")
    return int(row["n"])


async def _secondary_indexes(conn: psycopg.AsyncConnection, tables: list[str]) -> dict[str, str]:
    """Index name -> definition for the tables' indexes that do not back a constraint."""
    cur = await conn.cursor(row_factory=tuple_row).execute(
        "SELECT i.indexname, i.indexdef FROM pg_indexes i "
        "WHERE i.schemaname = current_schema() AND i.tablename = ANY(%s) AND NOT EXISTS ("
        "  SELECT 1 FROM pg_constraint c WHERE c.conname = i.indexname AND c.conrelid = (i.schemaname || '.' || i.tablename)::regclass)",
        (tables,))
    return dict(await cur.fetchall())


async def _load(source: Database, params: dict[str, Any], next_dsn: str) -> dict[str, int]:
    """Resumable: a table already loaded from the same number of source rows is not loaded again, and only
    missing indexes are built (each one is kept as soon as it is built)."""
    target = Database(next_dsn, 1, 1)
    await target.open()
    try:
        await migrate.migrate(target, params)
    finally:
        await target.close()
    wheres = {**{t: "" for t in TABLES}, EDGES: "WHERE status = 'accepted'"}
    async with source.tx() as src:
        indexes = await _secondary_indexes(src, list(wheres))
    async with await psycopg.AsyncConnection.connect(next_dsn, autocommit=True) as dst:
        await dst.execute(f"CREATE TABLE IF NOT EXISTS {PROGRESS_TABLE} (table_name text PRIMARY KEY, source_rows bigint NOT NULL)")
        cur = await dst.execute(f"SELECT table_name, source_rows FROM {PROGRESS_TABLE}")
        done = dict(await cur.fetchall())
        todo = {t for t, where in wheres.items() if done.get(t) != await _source_rows(source, t, where)}
        # A table referencing one that is reloaded is reloaded too: they are emptied together.
        cur = await dst.cursor(row_factory=tuple_row).execute(
            "SELECT conrelid::regclass::text, confrelid::regclass::text FROM pg_constraint WHERE contype = 'f'")
        references = await cur.fetchall()
        emptied = set(todo)  # with the tables not published that reference them (empty here)
        while grown := {child for child, parent in references if parent in emptied} - emptied:
            emptied |= grown
        todo = emptied & set(wheres)
        if todo:
            await dst.execute(sql.SQL("TRUNCATE {}").format(sql.SQL(", ").join(map(sql.Identifier, sorted(emptied)))))
            await dst.execute(f"DELETE FROM {PROGRESS_TABLE} WHERE table_name = ANY(%s)", (sorted(todo),))
        # Secondary indexes of the tables still to load are built once after the load, not row by row during it.
        present = await _secondary_indexes(dst, sorted(todo))
        for name in present:
            await dst.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(name)))
        for t in wheres:  # in TABLES order: a referenced table is loaded before the tables referencing it
            if t in todo:
                done[t] = await _copy(source, dst, t, wheres[t])
        built = await _secondary_indexes(dst, list(wheres))
        for name, definition in indexes.items():
            if name not in built:
                await dst.execute(definition)
        if todo:
            await dst.execute("ANALYZE")
    return {t: done[t] for t in wheres}


async def _carry_sign_ins(current_dsn: str, next_dsn: str) -> int:
    """The MCP server's OAuth clients and tokens, so publishing signs no one out."""
    async with await psycopg.AsyncConnection.connect(current_dsn, autocommit=True) as cur_db, \
            await psycopg.AsyncConnection.connect(next_dsn, autocommit=True) as dst:
        cur = await cur_db.execute("SELECT to_regclass(%s) IS NOT NULL", (OAUTH_TABLE,))
        if not (await cur.fetchone())[0]:
            return 0
        await dst.execute(OAUTH_SCHEMA)
        out = sql.SQL("COPY {} TO STDOUT (FORMAT BINARY)").format(sql.Identifier(OAUTH_TABLE))
        into = sql.SQL("COPY {} FROM STDIN (FORMAT BINARY)").format(sql.Identifier(OAUTH_TABLE))
        async with dst.transaction():  # taken fresh each time: sign-ins change while the graph loads
            await dst.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(OAUTH_TABLE)))
            async with cur_db.cursor().copy(out) as reader, dst.cursor().copy(into) as writer:
                async for block in reader:
                    await writer.write(block)
        cur = await dst.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(OAUTH_TABLE)))
        return (await cur.fetchone())[0]


async def _swap(admin: psycopg.AsyncConnection, name: str, nxt: str, prev: str, replace: bool) -> None:
    if not replace:
        await admin.execute(sql.SQL("ALTER DATABASE {} RENAME TO {}").format(sql.Identifier(nxt), sql.Identifier(name)))
        return
    await admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(prev)))
    await admin.execute(sql.SQL("ALTER DATABASE {} WITH ALLOW_CONNECTIONS false").format(sql.Identifier(name)))
    try:
        await admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()", (name,))
        await admin.execute(sql.SQL("ALTER DATABASE {} RENAME TO {}").format(sql.Identifier(name), sql.Identifier(prev)))
    finally:
        # Whichever name it has now, the replaced database stays usable as a rollback.
        current = prev if await _exists(admin, prev) else name
        await admin.execute(sql.SQL("ALTER DATABASE {} WITH ALLOW_CONNECTIONS true").format(sql.Identifier(current)))
    await admin.execute(sql.SQL("ALTER DATABASE {} RENAME TO {}").format(sql.Identifier(nxt), sql.Identifier(name)))


async def publish(source: Database, params: dict[str, Any], to: str, maintenance_db: str = "postgres") -> dict[str, Any]:
    conninfo = conninfo_to_dict(to)
    name = conninfo.get("dbname")
    if not name:
        raise PublishError("the target DSN names no database")
    nxt, prev = f"{name}_next", f"{name}_prev"
    admin_dsn = _dsn(conninfo, maintenance_db)
    async with await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True) as admin:
        replace = await _exists(admin, name)
        if not await _exists(admin, nxt):  # an earlier, interrupted publish is resumed
            await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(nxt)))
    next_dsn = _dsn(conninfo, nxt)
    counts = await _load(source, params, next_dsn)
    sign_ins = await _carry_sign_ins(_dsn(conninfo, name), next_dsn) if replace else 0
    async with await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True) as admin:
        await _swap(admin, name, nxt, prev, replace)
    return {"host": conninfo.get("host"), "database": name, "previous": prev if replace else None,
            "rows": counts, "mcp_sign_ins_carried": sign_ins}
