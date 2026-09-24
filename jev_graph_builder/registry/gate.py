"""The human gate on Registry proposals (§7.3 steps 5–6, §8.1 step 6, R-080).

`approve` records who approved a linted proposal; `activate` swaps the active
tree for an approved proposal and moves the corpus to its version, after which
dependency hashes make affected items stale (§14.3).
"""

from __future__ import annotations

from typing import Any

from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.registry.lint import lint_registry
from jev_graph_builder.registry.loader import Registry
from jev_graph_builder.registry.versioning import Proposal, activate as activate_tree
from jev_graph_builder.store import repo

APPROVED, ACTIVE, PROPOSED = "approved", "active", "proposed"
# `registry_versions.summary.kind`: which gate a proposal belongs to
BOOTSTRAP_KIND = "bootstrap"


class GateError(Exception):
    pass


async def record_proposal(ctx: Context, proposal: Proposal, summary: dict[str, Any]) -> None:
    async with ctx.db.tx() as conn:
        await repo.upsert(conn, "registry_versions", {"version": proposal.version, "corpus_id": ctx.corpus_id, "status": PROPOSED,
                                                      "summary": summary}, key=("version",), update=("summary",))


def lint_errors(proposal: Proposal) -> list[Any]:
    return lint_registry(Registry(proposal.path)).errors


async def record_approval(conn: Any, corpus_id: str, proposal: Proposal, summary: dict[str, Any], by: str) -> None:
    """Write an approved proposal row inside the caller's transaction."""
    await repo.upsert(conn, "registry_versions", {
        "version": proposal.version, "corpus_id": corpus_id, "status": APPROVED, "summary": summary,
        "approved_by": by}, key=("version",), update=("status", "summary", "approved_by"))
    await conn.execute("UPDATE registry_versions SET approved_at = now() WHERE version = %s", (proposal.version,))


async def approve(ctx: Context, proposal: Proposal, by: str, note: str | None = None) -> dict[str, Any]:
    errors = lint_errors(proposal)
    if errors:
        raise GateError(f"proposal {proposal.version} has lint errors: {errors}")
    async with ctx.db.tx() as conn:
        row = await repo.fetchone(conn, "SELECT summary FROM registry_versions WHERE version = %s", (proposal.version,))
        summary = {**((row or {}).get("summary") or {}), **({"approval_note": note} if note else {})}
        await record_approval(conn, ctx.corpus_id, proposal, summary, by)
    return {"version": proposal.version, "status": APPROVED, "approved_by": by}


async def activate(ctx: Context, proposal: Proposal) -> dict[str, Any]:
    row = await ctx.db.fetchone("SELECT status FROM registry_versions WHERE version = %s", (proposal.version,))
    if not row or row["status"] != APPROVED:
        raise GateError(f"proposal {proposal.version} is not approved (run `registry approve`)")
    previous = ctx.registry_version
    reg = activate_tree(ctx.reg.root, proposal)
    async with ctx.db.tx() as conn:
        await conn.execute("UPDATE registry_versions SET status = 'superseded' WHERE status = 'active' AND corpus_id = %s",
                           (ctx.corpus_id,))
        await conn.execute("UPDATE registry_versions SET status = 'active', activated_at = now() WHERE version = %s",
                           (proposal.version,))
        await conn.execute("UPDATE corpora SET registry_version = %s WHERE corpus_id = %s", (reg.version, ctx.corpus_id))
    return {"previous": previous, "active": reg.version}
