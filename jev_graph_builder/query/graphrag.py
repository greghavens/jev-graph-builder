"""Graph RAG (§9.2): Jev routes, code retrieves and expands (Jev-gated), Jev
decides the passages can answer. `context` stops there and returns the cited
passages (what a coding-agent session gets over MCP); `ask` goes on: a read-only
harness answers and Jev citation-checks every sentence."""

from __future__ import annotations

import asyncio
from typing import Any

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.jev.gating import ACCEPT, REJECT, top_level
from jev_graph_builder.jev.service import Decision
from jev_graph_builder.pipeline.citations import check_sentences
from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.pipeline.s6_links import relation_definitions
from jev_graph_builder.query.search import candidates, persist, rerank

QS_ROUTE, QS_EXPAND, QS_ANSWERABLE = "query_route", "expand_gate", "answerable"
PROMPT_ANSWER, SCHEMA_ANSWER = "answer", "answer"
ALL_RELATIONS = "*"
GROUNDING_OK, GROUNDING_LOW = "ok", "low"


async def route(ctx: Context, query: str) -> tuple[str, dict[str, Any], Decision]:
    pol = ctx.reg.policy("query")
    r = await ctx.jev.ask(QS_ROUTE, {"query": query, "corpus": {"scope": ctx.reg.corpus["scope"]}}, "query", sha256_hex(query))
    d = r.single
    intent = d.answers[pol["intent_question"]]["choice"]
    if d.outcome != ACCEPT:  # Jev said no: the policy's default plan
        intent = pol["fallback_intent"]
    plan = dict(ctx.reg.policy(f"retrieval_plans.{intent}"))
    overrides = plan.pop("by_complexity", None) or {}
    complexity = d.answers.get(pol["complexity_question"])
    if complexity and d.outcome == ACCEPT:
        plan.update(overrides.get(top_level(complexity), {}))
    return intent, plan, d



async def expand(ctx: Context, query: str, seeds: list[dict[str, Any]], plan: dict[str, Any]
                 ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[Decision]]:
    """Hop-by-hop expansion over accepted edges; each neighbour must pass `QS.expand_gate`."""
    rels = relation_definitions(ctx)
    allowed = plan["relations"]
    nodes = {s["chunk_id"]: s for s in seeds}
    used_edges: list[dict[str, Any]] = []
    decisions: list[Decision] = []
    frontier = list(nodes)
    sem = asyncio.Semaphore(ctx.reg.policy("run.concurrency.search"))
    for _hop in range(plan["max_hops"]):
        if not frontier:
            break
        rows = await ctx.db.fetch(
            "SELECT * FROM (SELECT e.edge_id, e.rel, e.weight, f.id AS from_id, CASE WHEN e.src_id = f.id THEN e.dst_id ELSE e.src_id END AS to_id, "
            " row_number() OVER (PARTITION BY f.id ORDER BY e.weight DESC NULLS LAST, e.edge_id) AS rn "
            " FROM unnest(%(f)s::text[]) AS f(id) JOIN edges e ON (e.src_id = f.id OR e.dst_id = f.id) "
            " WHERE e.status = 'accepted' AND NOT e.structural AND (%(any)s OR e.rel = ANY(%(rels)s))) x WHERE rn <= %(cap)s",
            {"f": frontier, "any": allowed == ALL_RELATIONS, "rels": [] if allowed == ALL_RELATIONS else list(allowed),
             "cap": plan["max_neighbors"]})
        rows = [r for r in rows if r["to_id"] not in nodes]
        texts = {r["chunk_id"]: r for r in await ctx.db.fetch(
            "SELECT c.chunk_id, c.doc_id, c.ord, c.title, c.text, c.heading_path, d.source_uri FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id WHERE c.chunk_id = ANY(%s) AND c.status = 'accepted' AND NOT coalesce(d.quarantined, false)",
            ([r["to_id"] for r in rows] + [r["from_id"] for r in rows],))}

        async def gate(r: dict[str, Any]) -> tuple[dict[str, Any], Decision | None]:
            src, dst = texts.get(r["from_id"]) or nodes.get(r["from_id"]), texts.get(r["to_id"])
            rel = rels.get(r["rel"])
            if dst is None or rel is None:
                return r, None
            async with sem:
                res = await ctx.jev.ask(QS_EXPAND, {
                    "query": query, "from_chunk": {"title": src.get("title") or "", "text": src.get("text") or ""},
                    "relation": {"name": rel["name"], "definition": rel["definition"]},
                    "neighbor": {"title": dst["title"] or "", "text": dst["text"]}}, "expansion", sha256_hex(query, r["edge_id"], r["to_id"]))
            return r, res.single

        added = []
        seen_to: set[str] = set()
        for r, d in await asyncio.gather(*(gate(r) for r in rows)):
            if d is None:
                continue
            decisions.append(d)
            if d.outcome == ACCEPT and r["to_id"] not in nodes and r["to_id"] not in seen_to:
                seen_to.add(r["to_id"])
                nodes[r["to_id"]] = dict(texts[r["to_id"]])
                used_edges.append({"edge_id": r["edge_id"], "src": r["from_id"], "dst": r["to_id"], "rel": r["rel"], "decision_id": d.decision_id})
                added.append(r["to_id"])
        frontier = added  # stop early when a hop adds nothing
    return nodes, used_edges, decisions


async def communities_for(ctx: Context, chunk_ids: list[str]) -> list[dict[str, Any]]:
    return await ctx.db.fetch(
        "SELECT community_id, summary FROM communities WHERE corpus_id = %s AND status = 'accepted' AND member_ids && %s "
        "ORDER BY community_id", (ctx.corpus_id, chunk_ids))


def assemble(ctx: Context, nodes: dict[str, dict[str, Any]], communities: list[dict[str, Any]], max_tokens: int) -> dict[str, dict[str, Any]]:
    """Dedupe, order by document then position, cap at the answer's token limit; assign citation IDs."""
    est = ctx.jev.provider.estimator
    ordered = sorted(nodes.values(), key=lambda n: (n.get("doc_id") or "", n.get("ord") or 0, n["chunk_id"]))
    passages: dict[str, dict[str, Any]] = {}
    used = 0
    items = [("chunk", n["chunk_id"], n["text"], n) for n in ordered] + [("community", c["community_id"], c["summary"], c) for c in communities]
    for kind, ident, text, src in items:
        tokens = est.text(text)
        if used + tokens > max_tokens:
            continue
        used += tokens
        cid = f"{ctx.reg.policy('query.citation_prefix')}{len(passages) + 1}"
        passages[cid] = {"kind": kind, "id": ident, "text": text, "source_uri": src.get("source_uri"), "doc_id": src.get("doc_id"),
                         "title": src.get("title"), "heading_path": src.get("heading_path")}
    return passages


async def answer(ctx: Context, query: str, plan: dict[str, Any], passages: dict[str, dict[str, Any]], feedback: list[dict[str, Any]] | None = None
                 ) -> tuple[dict[str, Any], str, str]:
    profile = plan.get("answer_profile") or await ctx.jobs.choose_profile(PROMPT_ANSWER, {"records": [query]})
    ws = ctx.jobs.workspace_for(PROMPT_ANSWER, sha256_hex(query, sorted(passages), feedback or []))
    ws.mkdir(parents=True, exist_ok=True)
    result, run_id = await ctx.jobs.run_single(PROMPT_ANSWER, SCHEMA_ANSWER, profile, {
        "query": query, "passages": [{"id": k, "text": v["text"]} for k, v in passages.items()], "failed_sentences": feedback or []}, ws)
    if not result.ok or not isinstance(result.structured, dict):
        raise RuntimeError(f"answer harness failed: {result.error}")
    return result.structured, run_id, profile


async def context(ctx: Context, query: str, max_tokens: int | None = None,
                  routed: tuple[str, dict[str, Any], Decision] | None = None) -> dict[str, Any]:
    """Route (unless `routed` is given), seed, expand and assemble the cited passages for `query`, capped at
    `max_tokens` (default: policy `query.context_tokens`). `answerable` is Jev's decision; when it is no, no passages
    are returned. The decisions made here are persisted here."""
    intent, plan, route_d = routed or await route(ctx, query)
    decisions: list[Decision] = [route_d]
    out: dict[str, Any] = {"query": query, "intent": intent, "plan": plan, "answerable": False, "passages": {},
                           "subgraph": {"nodes": [], "edges": []}}
    if plan.get("out_of_scope"):
        await persist(ctx, decisions)
        return {**out, "decisions": decisions}
    hits = await candidates(ctx, query)
    ranked, rerank_d = await rerank(ctx, query, hits)
    decisions.extend(rerank_d)
    seeds = [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "text": h.text, "title": h.title, "heading_path": h.heading_path,
              "source_uri": h.source_uri} for h in ranked[: plan["k_seeds"]]]
    nodes, edges, exp_d = await expand(ctx, query, seeds, plan)
    decisions.extend(exp_d)
    comms = await communities_for(ctx, list(nodes)) if plan.get("use_communities") else []
    passages = assemble(ctx, nodes, comms, max_tokens or ctx.reg.policy("query.context_tokens"))
    texts = {k: v["text"] for k, v in passages.items()}
    # Jev, not the harness or the session, decides whether the passages can answer the query.
    gate = await ctx.jev.ask(QS_ANSWERABLE, {"query": query, "passages": [{"id": k, "text": t} for k, t in texts.items()]},
                             "query", sha256_hex(query, sorted(texts)))
    decisions.append(gate.single)
    await persist(ctx, decisions)
    out["subgraph"] = {"nodes": sorted(nodes), "edges": edges}
    if texts and gate.single.outcome != REJECT:
        out.update(answerable=True, passages=passages)
    return {**out, "decisions": decisions}


def public(result: dict[str, Any]) -> dict[str, Any]:
    """A `context` result as returned to callers: decisions become their IDs, the internal plan is dropped."""
    out = {k: v for k, v in result.items() if k not in ("decisions", "plan")}
    out["decision_ids"] = [d.decision_id for d in result["decisions"]]
    return out


async def ask(ctx: Context, query: str) -> dict[str, Any]:
    pol = ctx.reg.policy("query")
    routed = await route(ctx, query)
    intent, plan = routed[0], routed[1]
    answer_tokens = ctx.reg.profile("harness", plan.get("answer_profile") or ctx.reg.policy(f"harness.defaults.{PROMPT_ANSWER}"))["context_tokens"]
    found = await context(ctx, query, answer_tokens, routed)
    nodes, edges = found["subgraph"]["nodes"], found["subgraph"]["edges"]
    decisions: list[Decision] = list(found["decisions"])
    passages = found["passages"]
    texts = {k: v["text"] for k, v in passages.items()}
    if not found["answerable"]:
        return {"query": query, "intent": intent, "unanswerable": True, "answer": [], "citations": {},
                "subgraph": found["subgraph"], "grounding": GROUNDING_OK, "decision_ids": [d.decision_id for d in decisions]}

    out, run_id, profile = await answer(ctx, query, plan, passages)
    runs = [run_id]
    persisted = len(decisions)
    check = await check_sentences(ctx.jev, out.get("answer_sentences", []), texts, ctx.reg.policy("run.concurrency.citation_check"), query)
    decisions.extend(check.decisions)
    if check.failed and pol["on_citation_fail"] == "regenerate":
        out2, run2, _ = await answer(ctx, query, plan, passages, feedback=check.failed)
        runs.append(run2)
        check2 = await check_sentences(ctx.jev, out2.get("answer_sentences", []), texts, ctx.reg.policy("run.concurrency.citation_check"), query)
        decisions.extend(check2.decisions)
        out, check = out2, check2
    grounding = GROUNDING_LOW if check.fail_fraction > pol["max_failed_fraction"] else GROUNDING_OK
    await persist(ctx, decisions[persisted:])
    cited = {c for s in check.kept for c in s.get("citation_ids") or []}
    return {
        "query": query, "intent": intent, "unanswerable": not check.kept,
        "answer": [{"text": s["text"], "citation_ids": s["citation_ids"], "decision_id": s["decision_id"]} for s in check.kept],
        "dropped_sentences": check.failed, "grounding": grounding,
        "citations": {k: v for k, v in passages.items() if k in cited},
        "subgraph": {"nodes": nodes, "edges": edges},
        "harness": {"profile": profile, "harness_run_ids": runs},
        "decision_ids": [d.decision_id for d in decisions],
    }
