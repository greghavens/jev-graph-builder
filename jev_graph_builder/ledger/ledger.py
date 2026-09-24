"""Run & work-item state machine (§14): pending → running → done | failed | review.

Items are skipped when a `done` row with the same input_hash exists. `running`
rows past their lease are reclaimed, so a crash anywhere is safe. The item's
graph rows, decisions and its `work_items` row commit in one transaction.

Leases: a live run heartbeats its `running` items, so a slow item (a long
harness job) never loses its lease. Each run also holds a session advisory lock
on a dedicated connection; when the process dies the server drops that lock, so
the dead run's items are reclaimed at once instead of after the full lease.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection

from jev_graph_builder.store import repo
from jev_graph_builder.store.db import Database

PENDING, RUNNING, DONE, FAILED, REVIEW = "pending", "running", "done", "failed", "review"


@dataclass(frozen=True)
class WorkItem:
    stage: str
    item_id: str
    input_hash: str
    payload: Any = None


class Ledger:
    def __init__(self, db: Database, lease_seconds: float, max_attempts: int, heartbeat_seconds: float | None = None) -> None:
        self.db = db
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.heartbeat_seconds = heartbeat_seconds
        self._owners: dict[str, tuple[AsyncConnection, asyncio.Task | None]] = {}

    async def start_run(self, stage: str, selector: dict, registry_version: str, config_hash: str) -> str:
        run_id = uuid.uuid4().hex
        owner = await AsyncConnection.connect(self.db.dsn, autocommit=True)
        await owner.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (run_id,))
        await self.db.execute(
            "INSERT INTO runs (run_id, stage, selector, registry_version, config_hash, status, started_at) "
            "VALUES (%s, %s, %s, %s, %s, 'running', now())",
            (run_id, stage, repo.Jsonb(selector), registry_version, config_hash),
        )
        beat = asyncio.create_task(self._heartbeat(owner, run_id)) if self.heartbeat_seconds else None
        self._owners[run_id] = (owner, beat)
        return run_id

    async def finish_run(self, run_id: str, status: str, report: dict) -> None:
        await self.db.execute(
            "UPDATE runs SET status = %s, ended_at = now(), report = %s WHERE run_id = %s", (status, repo.Jsonb(report), run_id)
        )
        owner, beat = self._owners.pop(run_id, (None, None))
        if beat is not None:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
        if owner is not None:
            await owner.close()

    async def _heartbeat(self, owner: AsyncConnection, run_id: str) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            await owner.execute("UPDATE work_items SET updated_at = now() WHERE run_id = %s AND status = 'running'", (run_id,))

    async def reclaim_expired(self) -> int:
        """Crash safety: `running` items go back to `pending` when their lease has
        expired or their run's owner lock is gone (the process died)."""
        rows = await self.db.fetch(
            "UPDATE work_items w SET status = 'pending' WHERE w.status = 'running' "
            "AND (w.updated_at < now() - make_interval(secs => %s) OR NOT EXISTS ("
            "  SELECT 1 FROM pg_locks l WHERE l.locktype = 'advisory' AND l.granted "
            "  AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database()) "
            "  AND ((l.classid::bigint << 32) | l.objid::bigint) = hashtextextended(w.run_id, 0))) "
            "RETURNING item_id",
            (self.lease_seconds,),
        )
        return len(rows)

    async def existing(self, stage: str) -> dict[str, dict]:
        rows = await self.db.fetch("SELECT item_id, input_hash, status, attempts FROM work_items WHERE stage = %s", (stage,))
        return {r["item_id"]: r for r in rows}

    @staticmethod
    def classify(item: WorkItem, row: dict | None) -> str:
        """missing | stale | failed | review | pending | done — used by the planner and the runner."""
        if row is None:
            return "missing"
        if row["input_hash"] != item.input_hash:
            return "stale"
        return row["status"]

    async def claim(self, item: WorkItem, run_id: str, registry_version: str) -> bool:
        """Atomically take the item unless another live worker holds it or it is already done."""
        rows = await self.db.fetch(
            "INSERT INTO work_items (stage, item_id, input_hash, registry_version, status, attempts, run_id, updated_at) "
            "VALUES (%(stage)s, %(item)s, %(hash)s, %(reg)s, 'running', 1, %(run)s, now()) "
            "ON CONFLICT (stage, item_id) DO UPDATE SET status = 'running', input_hash = EXCLUDED.input_hash, "
            "registry_version = EXCLUDED.registry_version, run_id = EXCLUDED.run_id, updated_at = now(), "
            "attempts = CASE WHEN work_items.input_hash = EXCLUDED.input_hash THEN work_items.attempts + 1 ELSE 1 END "
            "WHERE NOT (work_items.status = 'done' AND work_items.input_hash = EXCLUDED.input_hash) "
            "AND NOT (work_items.status = 'running' AND work_items.updated_at >= now() - make_interval(secs => %(lease)s)) "
            "AND NOT (work_items.status = 'failed' AND work_items.input_hash = EXCLUDED.input_hash AND work_items.attempts >= %(max)s) "
            "RETURNING item_id",
            {"stage": item.stage, "item": item.item_id, "hash": item.input_hash, "reg": registry_version, "run": run_id,
             "lease": self.lease_seconds, "max": self.max_attempts},
        )
        return bool(rows)

    async def heartbeat(self, item: WorkItem) -> None:
        await self.db.execute("UPDATE work_items SET updated_at = now() WHERE stage = %s AND item_id = %s", (item.stage, item.item_id))

    @staticmethod
    async def complete(conn: AsyncConnection, item: WorkItem, status: str = DONE) -> None:
        """Called inside the item's transaction so graph rows and ledger commit together."""
        await conn.execute(
            "UPDATE work_items SET status = %s, last_error = NULL, updated_at = now() WHERE stage = %s AND item_id = %s",
            (status, item.stage, item.item_id),
        )

    async def fail(self, item: WorkItem, error: str) -> str:
        rows = await self.db.fetch(
            "UPDATE work_items SET status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END, "
            "last_error = %s, updated_at = now() WHERE stage = %s AND item_id = %s RETURNING status",
            (self.max_attempts, error[:4000], item.stage, item.item_id),
        )
        return rows[0]["status"] if rows else FAILED

    async def release(self, item: WorkItem) -> None:
        """Run stop (e.g. Jev out of credits): leave the item pending (§16)."""
        await self.db.execute(
            "UPDATE work_items SET status = 'pending', attempts = greatest(attempts - 1, 0) WHERE stage = %s AND item_id = %s",
            (item.stage, item.item_id),
        )

    async def counts(self) -> list[dict]:
        return await self.db.fetch("SELECT stage, status, count(*) AS n FROM work_items GROUP BY stage, status ORDER BY stage, status")
