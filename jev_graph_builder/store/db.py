"""PostgreSQL access: one async connection pool, explicit transactions (R-030)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


class Database:
    def __init__(self, dsn: str, min_size: int, max_size: int) -> None:
        self.dsn = dsn
        self._pool = AsyncConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=False, configure=self._configure, kwargs={"row_factory": dict_row}
        )

    @staticmethod
    async def _configure(conn: AsyncConnection) -> None:
        try:
            await register_vector_async(conn)
        except Exception:  # extension not yet created (fresh DB before `init`)
            await conn.rollback()
        await conn.set_autocommit(False)
        await conn.commit()

    async def open(self) -> None:
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    async def reset_pool(self) -> None:
        """Re-register types after `CREATE EXTENSION vector` on a fresh database."""
        await self._pool.close()
        self._pool = AsyncConnectionPool(
            self.dsn, min_size=self._pool.min_size, max_size=self._pool.max_size, open=False,
            configure=self._configure, kwargs={"row_factory": dict_row},
        )
        await self.open()

    @asynccontextmanager
    async def tx(self) -> AsyncIterator[AsyncConnection]:
        """One ACID transaction: commits on success, rolls back on any exception."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                yield conn

    async def fetch(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        async with self.tx() as conn:
            cur = await conn.execute(sql, params)
            return list(await cur.fetchall()) if cur.description else []

    async def fetchone(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        rows = await self.fetch(sql, params)
        return rows[0] if rows else None

    async def execute(self, sql: str, params: Any = None) -> None:
        async with self.tx() as conn:
            await conn.execute(sql, params)


def jsonb(value: Any) -> Jsonb:
    return Jsonb(value)
