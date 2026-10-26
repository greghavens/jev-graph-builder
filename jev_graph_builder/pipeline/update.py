"""Bring a built graph up to date with new or changed documents, unattended (`jev-graph-builder update`)."""

from __future__ import annotations

from typing import Any

from jev_graph_builder.pipeline.common import Context, RunOptions, run_stage
from jev_graph_builder.pipeline.refresh_meta import refresh_metadata
from jev_graph_builder.pipeline.s1_ingest import IngestStage
from jev_graph_builder.pipeline.stages import graph_stages


async def update_graph(ctx: Context, docs: list[str]) -> dict[str, Any]:
    """S1 ingest of `docs` (a path already in the graph replaces its old version when its text changed), S2..S7,
    then the metadata refresh. Stops at the first stage that pauses or has failed items: `ok` is False then, nothing
    after it ran, and rerunning resumes. With no docs it only finishes what is still unbuilt."""
    stages = graph_stages()
    if docs:
        ctx.cache["ingest_paths"] = docs
    else:
        stages = [st for st in stages if not isinstance(st, IngestStage)]
    reports = []
    for st in stages:
        rep = await run_stage(ctx, st, RunOptions(), {"limit": None, "sample": None})
        reports.append(rep.as_dict())
        if rep.paused or rep.failed:
            return {"stages": reports, "ok": False}
    return {"stages": reports, "refresh": await refresh_metadata(ctx, True), "ok": True}
