"""Review queue for the QA spec's training stage (no graph stage writes to it).

A resolution is `{verdict: accept|reject, labels?, canonical_name?, note?}`.
Resolving applies the verdict to the subject row (in one transaction with the
queue row) and records whether it overturned Jev's decided outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from jev_graph_builder.store.db import Database
from jev_graph_builder.store.repo import Jsonb

OPEN, RESOLVED = "open", "resolved"
ACCEPT, REJECT = "accept", "reject"
VERDICT_STATUS = {ACCEPT: "accepted", REJECT: "rejected"}

# subject_kind -> (table, id column). Kinds absent here (entity pairs, gold items)
# have no row to update: their consumers read the resolution itself.
SUBJECT_TABLES = {
    "document": ("documents", "doc_id"),
    "chunk": ("chunks", "chunk_id"),
    "edge": ("edges", "edge_id"),
    "entity": ("entities", "entity_id"),
    "claim": ("claims", "claim_id"),
    "community": ("communities", "community_id"),
    "training_example": ("training_examples", "example_id"),
}
CLUSTER_KIND = "entity_cluster"


class ReviewError(Exception):
    pass


async def list_open(db: Database, reason: str | None = None, kind: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    sql = ["SELECT review_id, subject_kind, subject_id, reason, question_set, created_at FROM review_queue WHERE status = %s"]
    params: list[Any] = [OPEN]
    if reason:
        sql.append("AND reason = %s")
        params.append(reason)
    if kind:
        sql.append("AND subject_kind = %s")
        params.append(kind)
    sql.append("ORDER BY created_at, review_id")
    if limit:
        sql.append("LIMIT %s")
        params.append(limit)
    return await db.fetch(" ".join(sql), tuple(params))


async def show(db: Database, review_id: str) -> dict[str, Any]:
    row = await db.fetchone("SELECT * FROM review_queue WHERE review_id LIKE %s", (review_id + "%",))
    if row is None:
        raise ReviewError(f"no review `{review_id}`")
    if row["decision_id"]:
        row["decision"] = await db.fetchone("SELECT * FROM decisions WHERE decision_id = %s", (row["decision_id"],))
    table = SUBJECT_TABLES.get(row["subject_kind"])
    if table:
        subject = await db.fetchone(f"SELECT * FROM {table[0]} WHERE {table[1]} = %s", (row["subject_id"],))  # noqa: S608 (fixed map)
        if subject:
            subject.pop("embedding", None)
            subject.pop("fts", None)
        row["subject"] = subject
    return row


def is_overturn(decision_outcome: str | None, verdict: str) -> bool:
    """Only a *decided* outcome can be overturned; escalations are adjudications."""
    return decision_outcome in (ACCEPT, REJECT) and decision_outcome != verdict


async def resolve(db: Database, review_id: str, verdict: str, resolved_by: str, labels: dict[str, Any] | None = None,
                  canonical_name: str | None = None, note: str | None = None) -> dict[str, Any]:
    if verdict not in VERDICT_STATUS:
        raise ReviewError(f"verdict must be one of {sorted(VERDICT_STATUS)}")
    async with db.tx() as conn:
        cur = await conn.execute(
            "SELECT q.*, d.outcome AS decision_outcome FROM review_queue q LEFT JOIN decisions d ON d.decision_id = q.decision_id "
            "WHERE q.review_id LIKE %s FOR UPDATE OF q", (review_id + "%",))
        rows = await cur.fetchall()
        if len(rows) != 1:
            raise ReviewError(f"no unique review `{review_id}`")
        row = rows[0]
        if row["status"] == RESOLVED:
            raise ReviewError(f"review {row['review_id']} is already resolved")
        resolution: dict[str, Any] = {"verdict": verdict, "overturn": is_overturn(row["decision_outcome"], verdict)}
        if labels:
            resolution["labels"] = labels
        if canonical_name:
            resolution["canonical_name"] = canonical_name
        if note:
            resolution["note"] = note

        table = SUBJECT_TABLES.get(row["subject_kind"])
        if table:
            await conn.execute(f"UPDATE {table[0]} SET status = %s WHERE {table[1]} = %s",  # noqa: S608 (fixed map)
                               (VERDICT_STATUS[verdict], row["subject_id"]))
            if row["subject_kind"] == "document":
                # Accepting a quarantined document releases it; rejecting keeps it out of every stage.
                await conn.execute("UPDATE documents SET quarantined = %s, in_scope = %s WHERE doc_id = %s",
                                   (verdict == REJECT, verdict == ACCEPT, row["subject_id"]))
        if row["subject_kind"] == CLUSTER_KIND and canonical_name and verdict == ACCEPT:
            await conn.execute("UPDATE entities SET canonical_name = %s WHERE entity_id = %s", (canonical_name, row["subject_id"]))
        await conn.execute(
            "UPDATE review_queue SET status = %s, resolution = %s, resolved_by = %s, resolved_at = %s WHERE review_id = %s",
            (RESOLVED, Jsonb(resolution), resolved_by, datetime.now(UTC), row["review_id"]),
        )
    return {"review_id": row["review_id"], "subject_kind": row["subject_kind"], "subject_id": row["subject_id"], **resolution}


async def overturn_rates(db: Database) -> list[dict[str, Any]]:
    return await db.fetch(
        """
        SELECT question_set, count(*) AS resolved,
               count(*) FILTER (WHERE resolution->>'overturn' = 'true') AS overturned,
               round(count(*) FILTER (WHERE resolution->>'overturn' = 'true')::numeric / count(*), 4) AS rate
        FROM review_queue WHERE status = 'resolved' AND question_set IS NOT NULL GROUP BY question_set ORDER BY question_set
        """
    )
