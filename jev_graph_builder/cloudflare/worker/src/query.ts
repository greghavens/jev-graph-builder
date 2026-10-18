// The query path of jev_graph_builder/query (search.py, graphrag.py) over D1 and Vectorize.

import { ftsQuery } from "./fts";
import { sha256Hex } from "./hash";
import { ACCEPT, Decision, Jev, QuestionSet, topLevel } from "./jev";

export interface Bundle {
  registry_version: string;
  corpus_scope: string;
  question_sets: Record<string, QuestionSet>;
  policies: { query: Record<string, any>; retrieval_plans: Record<string, any>; search: Record<string, any>; concurrency: number };
  relations: Record<string, { name: string; definition: string }>;
  embedding: { workers_ai_model: string; pooling: string | null; dim: number };
}

const QS_ROUTE = "query_route", QS_PASSAGE = "rag_passage", QS_EXPAND = "expand_gate";
const NO_PEERS = `${QS_PASSAGE}.no_peers`;
const ALL_RELATIONS = "*";

export interface Hit {
  chunk_id: string; doc_id: string; source_uri: string; text: string; title: string | null; heading_path: string[];
  char_start: number | null; char_end: number | null; rrf: number; score: number; probabilities: Record<string, number>;
  decision_id: string | null;
}

interface ChunkRow { chunk_id: string; doc_id: string; ord: number; title: string | null; text: string; heading_path: string | null;
  source_uri: string; char_start: number | null; char_end: number | null }

function limiter(n: number) {
  let active = 0;
  const queue: (() => void)[] = [];
  return async <T>(fn: () => Promise<T>): Promise<T> => {
    if (active >= n) await new Promise<void>((r) => queue.push(r));
    active++;
    try {
      return await fn();
    } finally {
      active--;
      queue.shift()?.();
    }
  };
}

function rrfFuse(rankings: string[][], k: number): Map<string, number> {
  const scores = new Map<string, number>();
  for (const ranking of rankings) ranking.forEach((id, i) => scores.set(id, (scores.get(id) ?? 0) + 1 / (k + i + 1)));
  return scores;
}

const byKey = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

export class Graph {
  readonly decisions: Decision[] = [];
  private readonly limit: <T>(fn: () => Promise<T>) => Promise<T>;

  constructor(private readonly b: Bundle, private readonly jev: Jev, private readonly db: D1Database,
              private readonly vectors: VectorizeIndex, private readonly ai: Ai) {
    this.limit = limiter(b.policies.concurrency);
  }

  private async ask(name: string, inputs: Record<string, unknown>, kind: string, id: string): Promise<Decision> {
    const d = await this.limit(() => this.jev.ask(this.b.question_sets[name], inputs, kind, id));
    this.decisions.push(d);
    return d;
  }

  private async chunks(ids: string[]): Promise<Map<string, ChunkRow>> {
    const out = new Map<string, ChunkRow>();
    if (!ids.length) return out;
    const rows = await this.db.prepare(
      "SELECT chunk_id, doc_id, ord, title, text, heading_path, source_uri, char_start, char_end FROM chunks " +
      "WHERE chunk_id IN (SELECT value FROM json_each(?1))").bind(JSON.stringify([...new Set(ids)])).all<ChunkRow>();
    for (const r of rows.results) out.set(r.chunk_id, r);
    return out;
  }

  async route(query: string): Promise<{ intent: string; plan: Record<string, any> }> {
    const pol = this.b.policies.query;
    const d = await this.ask(QS_ROUTE, { query, corpus: { scope: this.b.corpus_scope } }, "query", await sha256Hex(query));
    const intent = d.outcome === ACCEPT ? d.answers[pol.intent_question].choice : pol.fallback_intent;
    const { by_complexity, ...plan } = this.b.policies.retrieval_plans[intent];
    const complexity = d.answers[pol.complexity_question];
    if (complexity && d.outcome === ACCEPT) Object.assign(plan, (by_complexity ?? {})[String(topLevel(complexity))] ?? {});
    return { intent, plan };
  }

  async candidates(query: string): Promise<Hit[]> {
    const pol = this.b.policies.search;
    const emb = this.b.embedding;
    const [vec, fts] = await Promise.all([
      this.ai.run(emb.workers_ai_model as any, { text: [query], ...(emb.pooling ? { pooling: emb.pooling } : {}) } as any)
        .then((r: any) => this.vectors.query(r.data[0], { topK: pol.k_ann })),
      (async () => {
        const match = ftsQuery(query);
        if (!match) return [] as string[];
        const rows = await this.db.prepare(
          "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.rid = f.rowid WHERE chunks_fts MATCH ?1 " +
          "ORDER BY bm25(chunks_fts), c.chunk_id LIMIT ?2").bind(match, pol.k_fts).all<{ chunk_id: string }>();
        return rows.results.map((r) => r.chunk_id);
      })(),
    ]);
    const ann = vec.matches.map((m) => m.id);
    const fused = rrfFuse([ann, fts], pol.rrf_k);
    const top = [...fused.keys()].sort((a, b) => fused.get(b)! - fused.get(a)! || byKey(a, b)).slice(0, pol.rerank_candidates);
    const rows = await this.chunks(top);
    return top.filter((id) => rows.has(id)).map((id) => {
      const r = rows.get(id)!;
      return { chunk_id: id, doc_id: r.doc_id, source_uri: r.source_uri, text: r.text, title: r.title,
               heading_path: r.heading_path ? JSON.parse(r.heading_path) : [], char_start: r.char_start, char_end: r.char_end,
               rrf: fused.get(id)!, score: 0, probabilities: {}, decision_id: null };
    });
  }

  async rerank(query: string, hits: Hit[]): Promise<Hit[]> {
    const pol = this.b.policies.search;
    const byRrf = [...hits].sort((a, b) => b.rrf - a.rrf || byKey(a.chunk_id, b.chunk_id));
    const peerIds = byRrf.map((p) => p.chunk_id).sort(byKey);
    await Promise.all(hits.map(async (h) => {
      // `contradicts_other_evidence` (§9.1): the passage is judged against the other top candidates.
      const peers = byRrf.filter((p) => p.chunk_id !== h.chunk_id).slice(0, pol.contradiction_peers).map((p) => p.text);
      const d = await this.ask(peers.length ? QS_PASSAGE : NO_PEERS,
        { query, passage: h.text, other_evidence: peers.length ? peers : null }, "query_passage",
        await sha256Hex(query, h.chunk_id, peerIds));
      h.decision_id = d.decision_id;
      h.probabilities = Object.fromEntries(Object.entries(d.answers).filter(([, a]) => a.type === "noul").map(([q, a]) => [q, a.p]));
      h.score = Object.entries(pol.weights as Record<string, number>).reduce((s, [q, w]) => s + (q in d.answers ? w * d.answers[q].p : 0), 0);
    }));
    return [...hits].sort((a, b) => b.score - a.score || b.rrf - a.rrf || byKey(a.chunk_id, b.chunk_id));
  }

  async expand(query: string, seeds: Record<string, any>[], plan: Record<string, any>) {
    const allowed = plan.relations;
    const nodes = new Map<string, Record<string, any>>(seeds.map((s) => [s.chunk_id, s]));
    const usedEdges: Record<string, string>[] = [];
    let frontier = [...nodes.keys()];
    for (let hop = 0; hop < plan.max_hops && frontier.length; hop++) {
      const edges = (await this.db.prepare(
        "SELECT * FROM (SELECT edge_id, rel, weight, from_id, to_id, row_number() OVER (PARTITION BY from_id " +
        "ORDER BY weight DESC NULLS LAST, edge_id) AS rn FROM (" +
        " SELECT e.edge_id, e.rel, e.weight, e.src_id AS from_id, e.dst_id AS to_id FROM edges e WHERE e.src_id IN (SELECT value FROM json_each(?1))" +
        " UNION ALL SELECT e.edge_id, e.rel, e.weight, e.dst_id, e.src_id FROM edges e WHERE e.dst_id IN (SELECT value FROM json_each(?1))" +
        ") WHERE ?2 OR rel IN (SELECT value FROM json_each(?3))) WHERE rn <= ?4")
        .bind(JSON.stringify(frontier), allowed === ALL_RELATIONS ? 1 : 0, JSON.stringify(allowed === ALL_RELATIONS ? [] : allowed),
              plan.max_neighbors).all<{ edge_id: string; rel: string; from_id: string; to_id: string }>()).results
        .filter((r) => !nodes.has(r.to_id));
      const texts = await this.chunks([...edges.map((r) => r.to_id), ...edges.map((r) => r.from_id)]);
      const gated = await Promise.all(edges.map(async (r) => {
        const src = texts.get(r.from_id) ?? nodes.get(r.from_id), dst = texts.get(r.to_id), rel = this.b.relations[r.rel];
        if (!dst || !rel || !src) return null;
        const d = await this.ask(QS_EXPAND, {
          query, from_chunk: { title: src.title ?? "", text: src.text ?? "" }, relation: { name: rel.name, definition: rel.definition },
          neighbor: { title: dst.title ?? "", text: dst.text } }, "expansion", await sha256Hex(query, r.edge_id, r.to_id));
        return { r, d };
      }));
      const added: string[] = [];
      for (const g of gated) {
        if (!g || g.d.outcome !== ACCEPT || nodes.has(g.r.to_id)) continue;
        const row = texts.get(g.r.to_id)!;
        nodes.set(g.r.to_id, { ...row, heading_path: row.heading_path ? JSON.parse(row.heading_path) : [] });
        usedEdges.push({ edge_id: g.r.edge_id, src: g.r.from_id, dst: g.r.to_id, rel: g.r.rel, decision_id: g.d.decision_id });
        added.push(g.r.to_id);
      }
      frontier = added; // stop early when a hop adds nothing
    }
    return { nodes, usedEdges };
  }

  async communities(chunkIds: string[]): Promise<{ community_id: string; summary: string }[]> {
    return (await this.db.prepare(
      "SELECT DISTINCT c.community_id, c.summary FROM community_members m JOIN communities c ON c.community_id = m.community_id " +
      "WHERE m.chunk_id IN (SELECT value FROM json_each(?1)) ORDER BY c.community_id").bind(JSON.stringify(chunkIds))
      .all<{ community_id: string; summary: string }>()).results;
  }

  assemble(nodes: Map<string, Record<string, any>>, comms: { community_id: string; summary: string }[]) {
    const ordered = [...nodes.values()].sort((a, b) =>
      byKey(a.doc_id ?? "", b.doc_id ?? "") || (a.ord ?? 0) - (b.ord ?? 0) || byKey(a.chunk_id, b.chunk_id));
    const items: [string, string, string, Record<string, any>][] = [
      ...ordered.map((n) => ["chunk", n.chunk_id, n.text, n] as [string, string, string, Record<string, any>]),
      ...comms.map((c) => ["community", c.community_id, c.summary, c] as [string, string, string, Record<string, any>])];
    const passages: Record<string, Record<string, unknown>> = {};
    items.forEach(([kind, id, text, src], i) => {
      passages[`${this.b.policies.query.citation_prefix}${i + 1}`] = { kind, id, text, source_uri: src.source_uri ?? null,
        doc_id: src.doc_id ?? null, title: src.title ?? null, heading_path: src.heading_path ?? null };
    });
    return passages;
  }

  /** graph_context: route, seed, expand and assemble the cited passages (no answerable gate). */
  async context(query: string) {
    const { intent, plan } = await this.route(query);
    const out = { query, intent, answerable: false, passages: {} as Record<string, unknown>,
                  subgraph: { nodes: [] as string[], edges: [] as Record<string, string>[] } };
    if (plan.out_of_scope) return out;
    const ranked = await this.rerank(query, await this.candidates(query));
    const seeds = ranked.slice(0, plan.k_seeds).map((h) => ({ chunk_id: h.chunk_id, doc_id: h.doc_id, text: h.text, title: h.title,
      heading_path: h.heading_path, source_uri: h.source_uri }));
    const { nodes, usedEdges } = await this.expand(query, seeds, plan);
    const comms = plan.use_communities ? await this.communities([...nodes.keys()]) : [];
    const passages = this.assemble(nodes, comms);
    out.subgraph = { nodes: [...nodes.keys()].sort(byKey), edges: usedEdges };
    if (Object.keys(passages).length) Object.assign(out, { answerable: true, passages });
    return out;
  }

  async search(query: string, k?: number) {
    const ranked = await this.rerank(query, await this.candidates(query));
    return { query, results: ranked.slice(0, k ?? this.b.policies.search.k) };
  }

  /** Query-time decisions are audited like pipeline ones (P3). */
  async persist(): Promise<string[]> {
    const now = new Date().toISOString();
    const stmt = this.db.prepare("INSERT OR REPLACE INTO decisions (decision_id, subject_kind, subject_id, question_set, outcome, answers, " +
      "created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)");
    if (this.decisions.length) {
      await this.db.batch(this.decisions.map((d) => stmt.bind(d.decision_id, d.subject_kind, d.subject_id, d.question_set, d.outcome,
        JSON.stringify({ answers: d.answers, threshold: d.threshold }), now)));
    }
    return this.decisions.map((d) => d.decision_id);
  }
}
