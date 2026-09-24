"""HTTP API (§9): `POST /search`. Runs with the reader role's DSN
when `JGB_READER_DSN` is set (§17 least privilege)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.query.search import search


class SearchRequest(BaseModel):
    query: str
    k: int | None = None
    filters: dict[str, Any] | None = None


def create_app(open_ctx: Callable[[], Awaitable[Context]], close_ctx: Callable[[Context], Awaitable[None]]) -> FastAPI:
    state: dict[str, Context] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["ctx"] = await open_ctx()
        try:
            yield
        finally:
            await close_ctx(state["ctx"])

    app = FastAPI(title="jev-graph-builder", lifespan=lifespan)

    @app.post("/search")
    async def post_search(req: SearchRequest) -> dict[str, Any]:
        return await search(state["ctx"], req.query, req.k, req.filters)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"registry_version": state["ctx"].registry_version, "jev_model": state["ctx"].jev.provider.model}

    return app
