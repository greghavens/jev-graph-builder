"""`jev-graph-builder status` / `audit graph` (§16) and the P3 audit query (§19 item 4).

The P3 rules (which rows are harness output, and which question sets count as
their verification) are Registry data: `policies.audit.p3_rules`, a list of
`{table, id, decisions_column, when, question_sets, subject_kind}`. `when` is a
SQL predicate naming the harness-produced columns (e.g. `summary IS NOT NULL`).
"""

from __future__ import annotations

from typing import Any

from jev_graph_builder.registry.loader import Registry
from jev_graph_builder.store.db import Database

_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyz_0123456789")


def _ident(name: str) -> str:
    """Registry-supplied identifiers are interpolated into SQL; accept plain names only."""
    if not name or set(name) - _IDENT_CHARS:
        raise ValueError(f"bad SQL identifier in p3 rule: {name!r}")
    return name


async def p3_violations(reg: Registry, db: Database, corpus_id: str, limit: int | None = None) -> dict[str, Any]:
    """Accepted harness-produced rows with no accepting Jev verification decision
    (or human acceptance of one). Zero is the acceptance bar."""
    out: dict[str, Any] = {"total": 0, "rules": {}}
    for rule in reg.policy("audit.p3_rules"):
        table, idc, dcol = _ident(rule["table"]), _ident(rule["id"]), _ident(rule["decisions_column"])
        scope = rule.get("corpus_join", "")
        sql = f"""
            SELECT t.{idc} AS id FROM {table} t {scope}
            WHERE t.status = 'accepted' AND ({rule['when']}) AND {rule['corpus_filter']}
              AND NOT EXISTS (
                SELECT 1 FROM decisions d
                WHERE d.decision_id = ANY(coalesce(t.{dcol}, ARRAY[]::text[]))
                  AND split_part(d.question_set, '@', 1) = ANY(%(qs)s)
                  AND (d.outcome = 'accept' OR EXISTS (
                        SELECT 1 FROM review_queue q WHERE q.decision_id = d.decision_id
                          AND q.status = 'resolved' AND q.resolution->>'verdict' = 'accept')))
            ORDER BY 1
        """  # noqa: S608 (identifiers validated; predicates are Registry-reviewed SQL)
        rows = await db.fetch(sql, {"qs": rule["question_sets"], "corpus": corpus_id})
        out["rules"][rule["name"]] = {"violations": len(rows), "examples": [r["id"] for r in rows[: limit or len(rows)]]}
        out["total"] += len(rows)
    return out


async def outcome_rates(db: Database, run_id: str | None = None) -> list[dict[str, Any]]:
    """Per question set: share accepted and rejected."""
    where, params = ("WHERE run_id = %s", (run_id,)) if run_id else ("", ())
    return await db.fetch(
        f"""
        SELECT question_set, count(*) AS decisions,
               round(avg((outcome = 'accept')::int), 4)::float AS accept,
               round(avg((outcome = 'reject')::int), 4)::float AS reject
        FROM decisions {where} GROUP BY question_set ORDER BY question_set
        """, params)  # noqa: S608


async def verification_pass_rate(reg: Registry, db: Database) -> dict[str, Any]:
    """Share of harness proposals that Jev verification accepted (§16)."""
    names = reg.policy("audit.verification_question_sets")
    rows = await db.fetch(
        """
        SELECT split_part(question_set, '@', 1) AS question_set, count(*) AS n,
               round(avg((outcome = 'accept')::int), 4)::float AS pass_rate
        FROM decisions WHERE split_part(question_set, '@', 1) = ANY(%s) GROUP BY 1 ORDER BY 1
        """, (names,))
    total = sum(r["n"] for r in rows)
    overall = sum(r["n"] * r["pass_rate"] for r in rows) / total if total else None
    return {"by_question_set": rows, "overall": overall}


async def stage_counts(db: Database) -> list[dict[str, Any]]:
    return await db.fetch("SELECT stage, status, count(*) AS n FROM work_items GROUP BY stage, status ORDER BY stage, status")


async def queue_sizes(db: Database) -> list[dict[str, Any]]:
    return await db.fetch(
        "SELECT reason, count(*) AS open FROM review_queue WHERE status = 'open' GROUP BY reason ORDER BY count(*) DESC, reason"
    )


async def graph_summary(db: Database, corpus_id: str) -> dict[str, Any]:
    edges = await db.fetch(
        "SELECT rel, structural, status, count(*) AS n FROM edges WHERE corpus_id = %s GROUP BY rel, structural, status "
        "ORDER BY structural, rel, status", (corpus_id,))
    counts = await db.fetchone(
        """
        SELECT (SELECT count(*) FROM documents WHERE corpus_id = %(c)s AND status <> 'superseded') AS documents,
               (SELECT count(*) FROM chunks WHERE corpus_id = %(c)s AND status <> 'superseded') AS chunks,
               (SELECT count(*) FROM entities WHERE corpus_id = %(c)s AND status = 'accepted' AND merged_into IS NULL) AS entities,
               (SELECT count(*) FROM communities WHERE corpus_id = %(c)s AND status = 'accepted') AS communities
        """, {"c": corpus_id})
    return {"counts": counts, "edges": edges}


async def alerts(db: Database, limit: int) -> list[dict[str, Any]]:
    return await db.fetch("SELECT kind, subject, payload, run_id, created_at FROM alerts ORDER BY created_at DESC LIMIT %s", (limit,))


async def edge_verification_violations(reg: Registry, db: Database, corpus_id: str) -> list[str]:
    """§19 item 3: every accepted semantic edge passed its required question sets, or a human accepted it.

    `policies.audit.edge_requirements` maps an edge's `src_kind` to groups of
    question sets; each group needs one accepting decision (e.g. chunk edges:
    `[[link, link_fanout], [link_verify]]`)."""
    requirements = reg.policy("audit.edge_requirements")
    rows = await db.fetch(
        """
        SELECT e.edge_id, e.src_kind,
               array_remove(array_agg(DISTINCT split_part(d.question_set, '@', 1)), NULL) AS passed
        FROM edges e
        LEFT JOIN decisions d ON d.decision_id = ANY(coalesce(e.decision_ids, ARRAY[]::text[])) AND d.outcome = 'accept'
        WHERE e.corpus_id = %s AND NOT e.structural AND e.status = 'accepted'
          AND NOT EXISTS (SELECT 1 FROM review_queue q WHERE q.subject_kind = 'edge' AND q.subject_id = e.edge_id
                          AND q.status = 'resolved' AND q.resolution->>'verdict' = 'accept')
        GROUP BY e.edge_id, e.src_kind ORDER BY e.edge_id
        """, (corpus_id,))
    bad = []
    for r in rows:
        groups = requirements.get(r["src_kind"])
        passed = set(r["passed"] or [])
        if groups is None or not all(passed & set(g) for g in groups):
            bad.append(r["edge_id"])
    return bad
