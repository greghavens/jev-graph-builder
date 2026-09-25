"""MCP server: the graph as context for a coding-agent session (Claude Code, Codex, OpenCode).

`graph_context` returns Graph RAG's cited passages (`query.graphrag.context`): Jev routes the
question, search seeds it, accepted edges are expanded through Jev's gate, and Jev decides the
passages can answer it. The session does its own answering. `search` is plain semantic search
(§9.1). Runs over stdio or streamable HTTP, with `JGB_READER_DSN` when set (§17 least privilege).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.mcpserver import MCPServer

from jev_graph_builder import log
from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.query.graphrag import context, public
from jev_graph_builder.query.search import search as do_search
from jev_graph_builder.registry.loader import Registry

SERVER_NAME = "jev-graph-builder"
_log = log.get(__name__)


def instructions_for(reg: Registry) -> str:
    """Policy `query.mcp_instructions` followed by the corpus scope, so the session knows which questions the graph covers."""
    return f"{reg.policy('query.mcp_instructions')}\n\n{reg.corpus['scope']}"


def create_server(instructions: str, open_ctx: Callable[[], Awaitable[Context]], close_ctx: Callable[[Context], Awaitable[None]]) -> MCPServer:
    """`instructions` tell the session when to call the tools (`instructions_for` builds them from the Registry)."""
    state: dict[str, Context] = {}

    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[None]:
        state["ctx"] = await open_ctx()
        try:
            yield
        finally:
            await close_ctx(state["ctx"])

    server = MCPServer(SERVER_NAME, instructions=instructions, lifespan=lifespan)

    @server.tool()
    async def graph_context(query: str) -> dict[str, Any]:
        """Cited passages from the documents' knowledge graph for a question: search seeds, then verified links
        between chunks are followed. Each passage has an ID (p1, p2, ...), text, title, heading path and source
        document. `answerable` is false (and `passages` empty) when the graph has nothing that answers it."""
        found = public(await context(state["ctx"], query))
        _log.info("mcp_graph_context", intent=found["intent"], answerable=found["answerable"],
                  passages=len(found["passages"]), decision_ids=found["decision_ids"])
        return found

    @server.tool()
    async def search(query: str, k: int | None = None) -> dict[str, Any]:
        """Ranked chunks for a query (semantic + full-text search, reranked), without following graph links."""
        found = await do_search(state["ctx"], query, k)
        _log.info("mcp_search", results=len(found["results"]))
        return found

    return server
