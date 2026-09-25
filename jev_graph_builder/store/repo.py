"""Plain-SQL repository helpers (R-030). Every write is an idempotent upsert
keyed by a deterministic ID, so re-running any stage is safe (P5)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Json, Jsonb

JSON_COLUMNS = {
    "meta", "features", "state", "answers", "usage", "payload", "resolution",
    "selector", "report", "questions", "keys", "dynamic",
}
# Columns whose name is jsonb only in some tables (`summary` is text on chunks/communities).
TABLE_JSON_COLUMNS = {"registry_versions": {"summary"}}
# `json` (not `jsonb`) columns: sent as text so key order survives (verbatim replay, §8.8).
ORDERED_JSON_COLUMNS = {"jev_calls": {"state", "questions", "dynamic"}}


def _adapt(table: str, col: str, value: Any) -> Any:
    if col in ORDERED_JSON_COLUMNS.get(table, ()):
        return Json(value) if value is not None and not isinstance(value, Json) else value
    is_json = col in JSON_COLUMNS or col in TABLE_JSON_COLUMNS.get(table, ())
    if is_json and value is not None and not isinstance(value, Jsonb):
        return Jsonb(value)
    return value


async def upsert(
    conn: AsyncConnection,
    table: str,
    row: dict[str, Any],
    key: Sequence[str],
    update: Iterable[str] | None = None,
) -> None:
    """INSERT … ON CONFLICT (key) DO UPDATE SET <update columns>.

    `update=None` updates every non-key column; `update=()` leaves an existing row untouched.
    """
    cols = list(row)
    upd = [c for c in (cols if update is None else update) if c not in key]
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) ON CONFLICT ({', '.join(key)}) "
    sql += f"DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in upd)}" if upd else "DO NOTHING"
    await conn.execute(sql, [_adapt(table, c, row[c]) for c in cols])


async def upsert_many(conn: AsyncConnection, table: str, rows: Sequence[dict[str, Any]], key: Sequence[str], update: Iterable[str] | None = None) -> None:
    for row in rows:
        await upsert(conn, table, row, key, update)


async def fetch(conn: AsyncConnection, sql: str, params: Any = None) -> list[dict[str, Any]]:
    cur = await conn.execute(sql, params)
    return list(await cur.fetchall()) if cur.description else []


async def fetchone(conn: AsyncConnection, sql: str, params: Any = None) -> dict[str, Any] | None:
    rows = await fetch(conn, sql, params)
    return rows[0] if rows else None


async def supersede(conn: AsyncConnection, table: str, id_col: str, ids: Sequence[str]) -> None:
    """Superseded rows are marked, never deleted (§14.3)."""
    if ids:
        await conn.execute(f"UPDATE {table} SET status = 'superseded' WHERE {id_col} = ANY(%s)", (list(ids),))


async def supersede_chunks(conn: AsyncConnection, ids: Sequence[str]) -> None:
    """Chunks superseded together with what was drawn from them: every edge that touches them, their mentions
    and claims, and each entity left with no mention outside superseded ones."""
    if ids:
        ids = list(ids)
        await supersede(conn, "chunks", "chunk_id", ids)
        await conn.execute("UPDATE edges SET status = 'superseded' WHERE src_id = ANY(%s) OR dst_id = ANY(%s)", (ids, ids))
        await conn.execute("UPDATE claims SET status = 'superseded' WHERE chunk_id = ANY(%s)", (ids,))
        await conn.execute("UPDATE mentions SET status = 'superseded' WHERE chunk_id = ANY(%s)", (ids,))
        await conn.execute(
            "UPDATE entities e SET status = 'superseded' WHERE e.entity_id IN (SELECT entity_id FROM mentions WHERE chunk_id = ANY(%s)) "
            "AND NOT EXISTS (SELECT 1 FROM mentions m WHERE m.entity_id = e.entity_id AND m.status <> 'superseded')", (ids,))
