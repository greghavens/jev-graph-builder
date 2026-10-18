"""OAuth for the MCP server over HTTPS, so web clients (claude.ai, ChatGPT) can connect.

The server is its own OAuth authorization server, as MCP clients expect: clients register themselves, the person
signs in with GitHub, and only the GitHub logins in `MCP_ALLOWED_GITHUB_USERS` get a token. Clients, pending logins,
codes and tokens are kept in Postgres (`mcp_oauth`), so a restart does not sign anyone out.

Environment: `MCP_PUBLIC_URL` (e.g. https://vcf-docs-mcp.fly.dev), `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`,
`MCP_ALLOWED_GITHUB_USERS` (comma-separated logins).
"""

from __future__ import annotations

import os
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"
CALLBACK_PATH = "/github/callback"
LOGIN_TTL_S = 600
CODE_TTL_S = 300
ACCESS_TTL_S = 24 * 3600
REFRESH_TTL_S = 90 * 24 * 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_oauth (
  kind text NOT NULL,
  key text NOT NULL,
  data jsonb NOT NULL,
  expires_at double precision,
  PRIMARY KEY (kind, key)
)"""


class GitHubOAuth:
    """`OAuthAuthorizationServerProvider` that signs people in with GitHub."""

    def __init__(self, dsn: str, public_url: str, client_id: str, client_secret: str, allowed: set[str]) -> None:
        self.pool = AsyncConnectionPool(dsn, min_size=1, max_size=4, open=False, check=AsyncConnectionPool.check_connection)
        self.public_url = public_url.rstrip("/")
        self.client_id, self.client_secret, self.allowed = client_id, client_secret, allowed

    async def open(self) -> None:
        await self.pool.open()
        async with self.pool.connection() as conn:
            await conn.execute(_SCHEMA)

    async def close(self) -> None:
        await self.pool.close()

    async def _put(self, kind: str, key: str, data: dict[str, Any], ttl: float | None) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO mcp_oauth (kind, key, data, expires_at) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (kind, key) DO UPDATE SET data = EXCLUDED.data, expires_at = EXCLUDED.expires_at",
                (kind, key, Jsonb(data), time.time() + ttl if ttl else None))

    async def _get(self, kind: str, key: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT data FROM mcp_oauth WHERE kind = %s AND key = %s AND (expires_at IS NULL OR expires_at > %s)",
                (kind, key, time.time()))).fetchone()
        return row[0] if row else None

    async def _pop(self, kind: str, key: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            row = await (await conn.execute(
                "DELETE FROM mcp_oauth WHERE kind = %s AND key = %s AND (expires_at IS NULL OR expires_at > %s) "
                "RETURNING data", (kind, key, time.time()))).fetchone()
        return row[0] if row else None

    # Clients register themselves (dynamic client registration).
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        data = await self._get("client", client_id)
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        await self._put("client", client_info.client_id or "", client_info.model_dump(mode="json"), None)

    # Authorization: send the person to GitHub; `callback` comes back with who they are.
    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        state = secrets.token_urlsafe(32)
        await self._put("login", state, {"client_id": client.client_id, "params": params.model_dump(mode="json")}, LOGIN_TTL_S)
        return f"{GITHUB_AUTHORIZE}?" + urlencode({
            "client_id": self.client_id, "redirect_uri": self.public_url + CALLBACK_PATH, "state": state,
            "scope": "read:user", "allow_signup": "false"})

    async def callback(self, request: Request) -> Response:
        login = await self._pop("login", request.query_params.get("state", ""))
        code = request.query_params.get("code")
        if not login or not code:
            return PlainTextResponse("Sign-in expired or invalid. Start again from your client.", status_code=400)
        async with httpx.AsyncClient(timeout=20) as http:
            tok = (await http.post(GITHUB_TOKEN, headers={"Accept": "application/json"}, data={
                "client_id": self.client_id, "client_secret": self.client_secret, "code": code,
                "redirect_uri": self.public_url + CALLBACK_PATH})).json()
            if "access_token" not in tok:
                return PlainTextResponse("GitHub sign-in failed.", status_code=400)
            user = (await http.get(GITHUB_USER, headers={"Authorization": f"Bearer {tok['access_token']}",
                                                         "Accept": "application/vnd.github+json"})).json()
        github_login = str(user.get("login", "")).lower()
        if github_login not in self.allowed:
            return PlainTextResponse(f"GitHub user {github_login!r} is not allowed on this server.", status_code=403)
        params = AuthorizationParams.model_validate(login["params"])
        auth_code = secrets.token_urlsafe(32)
        await self._put("code", auth_code, AuthorizationCode(
            code=auth_code, scopes=params.scopes or [], expires_at=time.time() + CODE_TTL_S, client_id=login["client_id"],
            code_challenge=params.code_challenge, redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly, resource=params.resource,
            subject=github_login).model_dump(mode="json"), CODE_TTL_S)
        return RedirectResponse(construct_redirect_uri(str(params.redirect_uri), code=auth_code, state=params.state),
                                status_code=302)

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str
                                      ) -> AuthorizationCode | None:
        data = await self._get("code", authorization_code)
        code = AuthorizationCode.model_validate(data) if data else None
        return code if code and code.client_id == client.client_id else None

    async def _issue(self, client_id: str, scopes: list[str], resource: str | None, subject: str | None) -> OAuthToken:
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        now = int(time.time())
        await self._put("access", access, AccessToken(token=access, client_id=client_id, scopes=scopes,
                                                      expires_at=now + ACCESS_TTL_S, resource=resource,
                                                      subject=subject).model_dump(mode="json"), ACCESS_TTL_S)
        await self._put("refresh", refresh, RefreshToken(token=refresh, client_id=client_id, scopes=scopes,
                                                         expires_at=now + REFRESH_TTL_S, resource=resource,
                                                         subject=subject).model_dump(mode="json"), REFRESH_TTL_S)
        return OAuthToken(access_token=access, token_type="Bearer", expires_in=ACCESS_TTL_S, refresh_token=refresh,
                          scope=" ".join(scopes) or None)

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
                                          ) -> OAuthToken:
        if not await self._pop("code", authorization_code.code):
            raise TokenError("invalid_grant", "authorization code already used or expired")
        return await self._issue(client.client_id or "", authorization_code.scopes, authorization_code.resource,
                                 authorization_code.subject)

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        data = await self._get("refresh", refresh_token)
        token = RefreshToken.model_validate(data) if data else None
        return token if token and token.client_id == client.client_id else None

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken,
                                     scopes: list[str]) -> OAuthToken:
        await self._pop("refresh", refresh_token.token)
        return await self._issue(client.client_id or "", scopes or refresh_token.scopes, refresh_token.resource,
                                 refresh_token.subject)

    async def load_access_token(self, token: str) -> AccessToken | None:
        data = await self._get("access", token)
        return AccessToken.model_validate(data) if data else None

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        await self._pop("access" if isinstance(token, AccessToken) else "refresh", token.token)


def from_env(dsn: str) -> GitHubOAuth | None:
    """The provider when `MCP_PUBLIC_URL` is set (a public HTTPS deployment), else None."""
    public_url = os.environ.get("MCP_PUBLIC_URL")
    if not public_url:
        return None
    missing = [k for k in ("GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "MCP_ALLOWED_GITHUB_USERS") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"MCP_PUBLIC_URL is set, so OAuth is required; missing {', '.join(missing)}")
    allowed = {u.strip().lower() for u in os.environ["MCP_ALLOWED_GITHUB_USERS"].split(",") if u.strip()}
    return GitHubOAuth(dsn, public_url, os.environ["GITHUB_CLIENT_ID"], os.environ["GITHUB_CLIENT_SECRET"], allowed)


def settings(provider: GitHubOAuth, mcp_path: str) -> AuthSettings:
    return AuthSettings(issuer_url=provider.public_url, resource_server_url=provider.public_url + mcp_path,
                        client_registration_options=ClientRegistrationOptions(enabled=True),
                        revocation_options=RevocationOptions(enabled=True))
