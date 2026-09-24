"""Numbered SQL migrations rendered from the embedding profile (§10.1, §14.5).

`vector(<dim>)` columns and HNSW parameters are *generated* from the
active embedding profile and policies at `init`, never hard-coded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.store.db import Database

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
_TABLE = "jgb_schema_migrations"


class MigrationError(Exception):
    pass


@dataclass(frozen=True)
class Migration:
    number: int
    name: str
    template: str


def available() -> list[Migration]:
    out = []
    for p in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _NAME.match(p.name)
        if m:
            out.append(Migration(int(m.group(1)), p.name, p.read_text(encoding="utf-8")))
    return out


def render(migration: Migration, params: dict) -> str:
    return Environment(undefined=StrictUndefined, autoescape=False).from_string(migration.template).render(**params)


async def _ensure_table(db: Database) -> None:
    await db.execute(
        f"CREATE TABLE IF NOT EXISTS {_TABLE} (number int PRIMARY KEY, name text, rendered_hash text, params jsonb, applied_at timestamptz DEFAULT now())"
    )


async def current_version(db: Database) -> int:
    await _ensure_table(db)
    row = await db.fetchone(f"SELECT coalesce(max(number), 0) AS n FROM {_TABLE}")
    return int(row["n"]) if row else 0


async def status(db: Database) -> tuple[int, int]:
    """(applied, latest known)."""
    return await current_version(db), max((m.number for m in available()), default=0)


async def migrate(db: Database, params: dict) -> list[str]:
    """Apply pending migrations; refuse when the database is ahead of this code (§14.5)."""
    from psycopg.types.json import Jsonb

    applied, latest = await status(db)
    if applied > latest:
        raise MigrationError(f"database schema version {applied} is ahead of this jev-graph-builder ({latest}); refusing to run")
    done = []
    for mig in available():
        if mig.number <= applied:
            continue
        sql = render(mig, params)
        async with db.tx() as conn:
            await conn.execute(sql)
            await conn.execute(
                f"INSERT INTO {_TABLE} (number, name, rendered_hash, params) VALUES (%s, %s, %s, %s)",
                (mig.number, mig.name, sha256_hex(sql), Jsonb(params)),
            )
        done.append(mig.name)
    if done:
        await db.reset_pool()
    return done


async def embedding_dim(db: Database) -> int | None:
    row = await db.fetchone(
        "SELECT atttypmod AS dim FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    )
    return int(row["dim"]) if row else None


async def reconcile_embedding_dim(db: Database, params: dict) -> bool:
    """When the embedding profile's dimension changes, retype the vector columns.

    Old vectors are dropped (set NULL); the embed stage's input_hash includes the
    embedding model ID, so the ledger re-embeds everything (§8.5, §14.3).
    """
    current = await embedding_dim(db)
    if current is None or current == params["dim"]:
        return False
    dim = int(params["dim"])
    async with db.tx() as conn:
        for table in ("chunks", "entities", "claims"):
            await conn.execute(f"DROP INDEX IF EXISTS {table}_embedding_hnsw")
            await conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({dim}) USING NULL")
            await conn.execute(
                f"CREATE INDEX {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {int(params['hnsw_m'])}, ef_construction = {int(params['hnsw_ef_construction'])})"
            )
            await conn.execute(f"UPDATE {table} SET embedding_model_id = NULL")
    await db.reset_pool()
    return True
