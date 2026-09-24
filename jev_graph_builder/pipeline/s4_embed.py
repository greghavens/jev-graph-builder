"""S4: embeddings and indexing (§8.5).

Chunks are embedded with their heading-path prefix (template is Registry
data), entities as their extracted surface name + description, claims as text.
Entity text never uses S5's output (canonical names, merges): resolution
compares surface forms, and re-embedding after S5 would change the candidate
pairs, so a rerun would not be idempotent. Every vector
stores `embedding_model_id`; a model change alters the input hash, so the
ledger re-embeds exactly what the old model produced (§14.3). The HNSW and
full-text indexes are maintained by PostgreSQL; `finalize` refreshes planner
statistics.
"""

from __future__ import annotations

from typing import Any

import jinja2

from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.pipeline.common import Context, Deps, Stage, Writer

CHUNK, ENTITY, CLAIM = "chunk", "entity", "claim"
_SEP = ":"
CACHE_KEY = "s4_vectors"


def item_key(kind: str, row_id: str) -> str:
    return f"{kind}{_SEP}{row_id}"


class EmbedStage(Stage):
    name = "embed"
    deps = Deps(policies=("embed",), embedding=True)

    def _templates(self, ctx: Context) -> dict[str, jinja2.Template]:
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
        return {k: env.from_string(ctx.reg.policy(f"embed.templates.{k}")) for k in (CHUNK, ENTITY, CLAIM)}

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        tpl = self._templates(ctx)
        items: list[WorkItem] = []
        chunks = await ctx.db.fetch(
            "SELECT chunk_id, text, heading_path, title FROM chunks WHERE corpus_id = %s AND status = 'accepted' ORDER BY chunk_id",
            (ctx.corpus_id,))
        for r in chunks:
            text = tpl[CHUNK].render(heading_path=r["heading_path"] or [], text=r["text"], title=r["title"] or "")
            items.append(self.item(ctx, item_key(CHUNK, r["chunk_id"]), text, payload=text))
        entities = await ctx.db.fetch(
            "SELECT entity_id, coalesce(surface, canonical_name) AS name, entity_type, description FROM entities "
            "WHERE corpus_id = %s AND status = 'accepted' ORDER BY entity_id", (ctx.corpus_id,))
        for r in entities:
            text = tpl[ENTITY].render(name=r["name"], type=r["entity_type"], description=r["description"] or "")
            items.append(self.item(ctx, item_key(ENTITY, r["entity_id"]), text, payload=text))
        claims = await ctx.db.fetch(
            "SELECT cl.claim_id, cl.text FROM claims cl JOIN chunks c ON c.chunk_id = cl.chunk_id "
            "WHERE c.corpus_id = %s AND cl.status = 'accepted' ORDER BY cl.claim_id", (ctx.corpus_id,))
        for r in claims:
            text = tpl[CLAIM].render(text=r["text"])
            items.append(self.item(ctx, item_key(CLAIM, r["claim_id"]), text, payload=text))
        return items

    async def prepare(self, ctx: Context, items: list[WorkItem], enumerated: list[WorkItem]) -> None:
        size = ctx.reg.policy("embed.batch_size")
        cache: dict[str, list[float]] = ctx.cache.setdefault(CACHE_KEY, {})
        for start in range(0, len(items), size):
            batch = items[start: start + size]
            vectors = await ctx.embedder.embed([i.payload for i in batch])
            cache.update({i.item_id: v for i, v in zip(batch, vectors, strict=True)})

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        vector = ctx.cache.get(CACHE_KEY, {}).pop(item.item_id, None)
        if vector is None:
            vector = (await ctx.embedder.embed([item.payload]))[0]
        kind, row_id = item.item_id.split(_SEP, 1)
        model_id = ctx.embedder.model_id
        sql: dict[str, tuple[str, tuple[Any, ...]]] = {
            CHUNK: ("UPDATE chunks SET embedding = %s, embedding_model_id = %s WHERE chunk_id = %s", (vector, model_id, row_id)),
            ENTITY: ("UPDATE entities SET embedding = %s, embedding_model_id = %s WHERE entity_id = %s", (vector, model_id, row_id)),
            CLAIM: ("UPDATE claims SET embedding = %s, embedding_model_id = %s WHERE claim_id = %s", (vector, model_id, row_id)),
        }
        query, params = sql[kind]

        async def write(conn) -> str:
            await conn.execute(query, (_vec(params[0]), *params[1:]))
            return DONE

        return write

    async def finalize(self, ctx: Context, report: Any, opts: Any) -> None:
        async with ctx.db.tx() as conn:
            for table in ("chunks", "entities", "claims"):
                await conn.execute(f"ANALYZE {table}")


def _vec(v: list[float]) -> Any:
    import numpy as np

    return np.asarray(v, dtype=np.float32)
