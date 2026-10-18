// Sign-in with GitHub (as jev_graph_builder/mcp_auth.py): only the allowed GitHub users get a token for /mcp.

import type { AuthRequest, OAuthHelpers } from "@cloudflare/workers-oauth-provider";

export interface AuthEnv {
  OAUTH_KV: KVNamespace;
  OAUTH_PROVIDER: OAuthHelpers;
  GITHUB_CLIENT_ID: string;
  GITHUB_CLIENT_SECRET: string;
  ALLOWED_GITHUB_USERS: string;
}

const LOGIN_TTL_S = 600;
const LOGIN_PREFIX = "login:";

export async function handle(request: Request, env: AuthEnv): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/authorize") {
    const authRequest = await env.OAUTH_PROVIDER.parseAuthRequest(request);
    const state = crypto.randomUUID();
    await env.OAUTH_KV.put(LOGIN_PREFIX + state, JSON.stringify(authRequest), { expirationTtl: LOGIN_TTL_S });
    const github = new URL("https://github.com/login/oauth/authorize");
    github.search = new URLSearchParams({ client_id: env.GITHUB_CLIENT_ID, redirect_uri: `${url.origin}/callback`,
                                          scope: "read:user", state }).toString();
    return Response.redirect(github.toString(), 302);
  }
  if (url.pathname === "/callback") {
    const state = url.searchParams.get("state") ?? "";
    const code = url.searchParams.get("code");
    const stored = await env.OAUTH_KV.get(LOGIN_PREFIX + state);
    if (!stored || !code) return new Response("Sign-in expired or invalid. Start again from your client.", { status: 400 });
    await env.OAUTH_KV.delete(LOGIN_PREFIX + state);
    const token = (await (await fetch("https://github.com/login/oauth/access_token", {
      method: "POST", headers: { accept: "application/json", "content-type": "application/json" },
      body: JSON.stringify({ client_id: env.GITHUB_CLIENT_ID, client_secret: env.GITHUB_CLIENT_SECRET, code,
                             redirect_uri: `${url.origin}/callback` }),
    })).json()) as { access_token?: string };
    if (!token.access_token) return new Response("GitHub sign-in failed.", { status: 400 });
    const user = (await (await fetch("https://api.github.com/user", {
      headers: { authorization: `Bearer ${token.access_token}`, accept: "application/vnd.github+json", "user-agent": "jev-graph-builder" },
    })).json()) as { login?: string };
    const login = String(user.login ?? "").toLowerCase();
    const allowed = new Set(env.ALLOWED_GITHUB_USERS.split(",").map((u) => u.trim().toLowerCase()).filter(Boolean));
    if (!allowed.has(login)) return new Response(`GitHub user '${login}' is not allowed on this server.`, { status: 403 });
    const authRequest = JSON.parse(stored) as AuthRequest;
    const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
      request: authRequest, userId: login, metadata: { label: login }, scope: authRequest.scope, props: { login },
    });
    return Response.redirect(redirectTo, 302);
  }
  return new Response("Not found", { status: 404 });
}
