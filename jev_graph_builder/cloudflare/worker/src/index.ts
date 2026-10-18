// The graph's MCP server on Cloudflare: `graph_context` and `search` (as jev_graph_builder/query/mcp_server.py),
// behind OAuth with GitHub sign-in. `bundle.json` is written by `jev-graph-builder publish --target cloudflare`.

import { OAuthProvider } from "@cloudflare/workers-oauth-provider";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";

import bundleJson from "./bundle.json";
import { AuthEnv, handle } from "./github";
import { Estimator, Jev, Retry, SystemOne, systemOneCaller } from "./jev";
import { Bundle, Graph } from "./query";

interface Env extends AuthEnv {
  DB: D1Database;
  VECTORS: VectorizeIndex;
  AI: Ai;
  SYSTEM_ONE_KEY?: string;
  SYSTEM_ONE_BINDING: string;
}

const bundle = bundleJson as unknown as Bundle & {
  server: { name: string; instructions: string };
  chars_per_token: number;
  untrusted_field: string;
  system_one: SystemOne;
  retry: Retry;
};

function graph(env: Env): Graph {
  const call = systemOneCaller(bundle.system_one, bundle.retry, env.SYSTEM_ONE_KEY,
                               env.SYSTEM_ONE_BINDING === "workers_ai" ? env.AI : null);
  const jev = new Jev(call, bundle.system_one, new Estimator(bundle.chars_per_token), bundle.untrusted_field);
  return new Graph(bundle, jev, env.DB, env.VECTORS, env.AI);
}

function result(value: Record<string, unknown>) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value) }], structuredContent: value };
}

function server(env: Env): McpServer {
  const mcp = new McpServer({ name: bundle.server.name, version: bundle.registry_version },
                            { instructions: bundle.server.instructions });
  mcp.registerTool("graph_context", {
    description: "Cited passages from the documents' knowledge graph for a question: search seeds, then verified links between " +
      "chunks are followed. Each passage has an ID (p1, p2, ...), text, title, heading path and source document. `answerable` is " +
      "false (and `passages` empty) only when nothing was found; judge relevance yourself.",
    inputSchema: { query: z.string() },
  }, async ({ query }) => {
    const g = graph(env);
    const found = await g.context(query);
    return result({ ...found, decision_ids: await g.persist() });
  });
  mcp.registerTool("search", {
    description: "Ranked chunks for a query (semantic + full-text search, reranked), without following graph links.",
    inputSchema: { query: z.string(), k: z.number().int().optional() },
  }, async ({ query, k }) => {
    const g = graph(env);
    const found = await g.search(query, k);
    return result({ ...found, decision_ids: await g.persist() });
  });
  return mcp;
}

const mcpHandler = {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Stateless: a server and transport per request, so any isolate can answer any request.
    const transport = new WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    await server(env).connect(transport);
    return transport.handleRequest(request);
  },
};

let provider: OAuthProvider<Env> | undefined;

export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    // The public URL is this Worker's own, known from the first request.
    const origin = new URL(request.url).origin;
    provider ??= new OAuthProvider<Env>({
      apiRoute: "/mcp",
      apiHandler: mcpHandler as any,
      defaultHandler: { fetch: (req: Request, e: Env) => handle(req, e) } as any,
      authorizeEndpoint: "/authorize",
      tokenEndpoint: "/token",
      clientRegistrationEndpoint: "/register",
      resourceMetadata: { resource: `${origin}/mcp`, authorization_servers: [origin], resource_name: bundle.server.name },
    });
    return provider.fetch(request, env, ctx);
  },
};
