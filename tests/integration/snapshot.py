"""Canonical dump of the final graph state, for the byte-identical resume check (§19).

Timestamps and run IDs are excluded, as the spec allows. So is purely
operational bookkeeping that records *how* a run went rather than what it
produced: the ledger itself, run rows, and per-attempt harness/Jev telemetry.
"""

from __future__ import annotations

import hashlib
import json

import psycopg

# Operational tables: their row counts legitimately depend on where a run was killed.
SKIP_TABLES = {"runs", "work_items", "harness_runs", "alerts", "jgb_schema_migrations", "registry_versions"}
# Columns that are timestamps, run identifiers, or per-attempt telemetry.
SKIP_COLUMNS = {"run_id", "created_run_id", "harness_run_id", "harness_run_ids", "request_id", "latency_ms"}
# Drift samples are drawn per run (salted with the run ID, §8.8): audit telemetry, not graph state.
ROW_FILTERS = {"jev_calls": "NOT drift"}
SKIP_TYPES = {"timestamp with time zone", "timestamp without time zone"}


def snapshot(dsn: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with psycopg.connect(dsn) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY 1")]
        for table in tables:
            if table in SKIP_TABLES:
                continue
            cols = [r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s "
                "AND data_type <> ALL(%s) ORDER BY ordinal_position", (table, list(SKIP_TYPES)))
                if r[0] not in SKIP_COLUMNS]
            select = ", ".join(f'"{c}"::text' for c in cols)
            where = f" WHERE {ROW_FILTERS[table]}" if table in ROW_FILTERS else ""
            rows = [json.dumps(dict(zip(cols, r, strict=True)), ensure_ascii=False, sort_keys=True)
                    for r in conn.execute(f'SELECT {select} FROM "{table}"{where}')]  # noqa: S608
            out[table] = sorted(rows)
    return out


def digest(snap: dict[str, list[str]]) -> str:
    return hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest()


def diff(a: dict[str, list[str]], b: dict[str, list[str]], limit: int = 3) -> dict[str, dict[str, list[str]]]:
    out = {}
    for t in sorted(set(a) | set(b)):
        x, y = set(a.get(t, [])), set(b.get(t, []))
        if x != y:
            out[t] = {"only_first": sorted(x - y)[:limit], "only_second": sorted(y - x)[:limit],
                      "counts": [len(a.get(t, [])), len(b.get(t, []))]}
    return out
