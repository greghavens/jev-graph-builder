"""`jev-graph-builder publish --target cloudflare`: serve the graph from a Cloudflare Worker.

The graph goes into a new D1 database (chunks with full-text search, accepted edges, communities) and a new
Vectorize index (the chunks embedded with the serving embedding profile, which Workers AI also runs for the
question), both named after the data they hold. The Worker (`worker/`, this package's TypeScript) is then
deployed pointing at them, so switching is one deploy; the previous database and index are kept.

Every step is recorded in `<workspace>/cloudflare/<name>/` as it finishes, so a rerun resumes: SQL files
and embedding files already written are reused, files already imported or upserted are skipped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2

from jev_graph_builder import log
from jev_graph_builder.cloudflare import bundle
from jev_graph_builder.embed.base import build_embedder
from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.pipeline.common import Context

WORKER_SRC = Path(__file__).parent / "worker"
STATE_FILE = "state.json"
SCHEMA_FILE = "schema.sql"
OAUTH_KV = "OAUTH_KV"
ALLOWED_USERS_ENV = "MCP_ALLOWED_GITHUB_USERS"
# Worker secrets; the System One key is only needed when it is not called through the Workers AI binding.
SECRETS = ("GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET")
SYSTEM_ONE_KEY = "SYSTEM_ONE_KEY"
WORKERS_DEV_URL = re.compile(r"https://\S+\.workers\.dev")
WORKERS_AI_RUN = re.compile(r"^https://api\.cloudflare\.com/client/v4/accounts/[^/]+/ai/run/(?P<model>.+)$")

SCHEMA = """CREATE TABLE IF NOT EXISTS chunks (rid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, doc_id TEXT, ord INTEGER,
  title TEXT, text TEXT, heading_path TEXT, role TEXT, doc_type TEXT, source_uri TEXT, char_start INTEGER, char_end INTEGER, meta TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, tokenize = 'unicode61');
CREATE TABLE IF NOT EXISTS edges (edge_id TEXT PRIMARY KEY, src_id TEXT NOT NULL, dst_id TEXT NOT NULL, rel TEXT, weight REAL);
CREATE INDEX IF NOT EXISTS edges_src ON edges (src_id, weight);
CREATE INDEX IF NOT EXISTS edges_dst ON edges (dst_id, weight);
CREATE TABLE IF NOT EXISTS communities (community_id TEXT PRIMARY KEY, summary TEXT);
CREATE TABLE IF NOT EXISTS community_members (community_id TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY (community_id, chunk_id));
CREATE INDEX IF NOT EXISTS community_members_chunk ON community_members (chunk_id);
CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT PRIMARY KEY, subject_kind TEXT, subject_id TEXT, question_set TEXT,
  outcome TEXT, answers TEXT, created_at TEXT);
"""

# What the query path serves: accepted chunks of documents not quarantined (as in query/search.py).
CHUNKS_SQL = """SELECT c.chunk_id, c.doc_id, c.ord, c.title, c.text, c.heading_path, c.role, d.doc_type, d.source_uri, c.meta,
  u1.char_start, u2.char_end FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
  LEFT JOIN units u1 ON u1.unit_id = c.unit_ids[1] LEFT JOIN units u2 ON u2.unit_id = c.unit_ids[cardinality(c.unit_ids)]
  WHERE c.corpus_id = %s AND c.status = 'accepted' AND NOT coalesce(d.quarantined, false) ORDER BY c.chunk_id"""
EDGES_SQL = """SELECT edge_id, src_id, dst_id, rel, weight FROM edges WHERE corpus_id = %s AND status = 'accepted' AND NOT structural
  ORDER BY edge_id"""
COMMUNITIES_SQL = """SELECT community_id, summary, member_ids FROM communities WHERE corpus_id = %s AND status = 'accepted'
  ORDER BY community_id"""


class CloudflareError(Exception):
    pass


def _sql(value: Any) -> str:
    """A SQLite literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return "'" + str(value).replace("'", "''") + "'"


def _insert(table: str, row: dict[str, Any]) -> str:
    return f"INSERT OR REPLACE INTO {table} ({', '.join(row)}) VALUES ({', '.join(_sql(v) for v in row.values())});\n"


class _State:
    """A JSON file updated as each step finishes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def add(self, key: str, item: str) -> None:
        self.set(key, sorted({*(self[key] or []), item}))


class Wrangler:
    def __init__(self, worker_dir: Path) -> None:
        self.cmd = [str(worker_dir / "node_modules" / ".bin" / "wrangler")]
        self.cwd = worker_dir

    def __call__(self, *args: str, check: bool = True) -> str:
        done = subprocess.run([*self.cmd, *args], cwd=self.cwd, capture_output=True, text=True)
        if check and done.returncode:
            raise CloudflareError(f"wrangler {' '.join(args[:3])} failed: {(done.stderr or done.stdout).strip()[-2000:]}")
        return done.stdout

    def json(self, *args: str) -> Any:
        return json.loads(self(*args, "--json"))


def _worker_dir(root: Path) -> Path:
    """This package's Worker source, copied into the build directory, with its npm dependencies installed."""
    work = root / "worker"
    if work.exists():
        shutil.rmtree(work / "src", ignore_errors=True)
    shutil.copytree(WORKER_SRC, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns("node_modules"))
    lock = work / "package-lock.json"
    stamp = work / "node_modules" / ".installed"
    wanted = sha256_hex((work / "package.json").read_text(), lock.read_text() if lock.exists() else "")
    if not stamp.exists() or stamp.read_text() != wanted:
        npm = shutil.which("npm")
        if npm is None:
            raise CloudflareError("npm is needed to deploy the Cloudflare Worker")
        done = subprocess.run([npm, "ci" if lock.exists() else "install", "--no-audit", "--no-fund"], cwd=work,
                              capture_output=True, text=True)
        if done.returncode:
            raise CloudflareError(f"npm install failed: {done.stderr.strip()[-2000:]}")
        stamp.write_text(wanted)
    return work


async def _fingerprint(ctx: Context, serving: dict[str, Any]) -> str:
    """The data this version serves: which chunks, accepted edges and communities, and the serving embedding."""
    row = await ctx.db.fetchone(
        f"SELECT (SELECT md5(string_agg(chunk_id, ',' ORDER BY chunk_id)) FROM ({CHUNKS_SQL}) c) AS chunks, "
        f"(SELECT md5(string_agg(edge_id, ',' ORDER BY edge_id)) FROM ({EDGES_SQL}) e) AS edges, "
        f"(SELECT md5(string_agg(community_id, ',' ORDER BY community_id)) FROM ({COMMUNITIES_SQL}) m) AS communities",
        (ctx.corpus_id, ctx.corpus_id, ctx.corpus_id))
    return sha256_hex(row, serving["model_id"], SCHEMA)


async def _export_sql(ctx: Context, out: Path, file_bytes: int) -> list[str]:
    """SQL files for D1 (one INSERT per row, under its statement limit), written to `.tmp` and renamed when complete."""
    (out / SCHEMA_FILE).write_text(SCHEMA)
    names: list[str] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            name = f"data-{len(names):04d}.sql"
            tmp = out / (name + ".tmp")
            tmp.write_text("".join(buf))
            tmp.replace(out / name)
            names.append(name)
            buf, size = [], 0

    def emit(stmt: str) -> None:
        nonlocal size
        buf.append(stmt)
        size += len(stmt.encode())
        if size >= file_bytes:
            flush()

    for rid, r in enumerate(await ctx.db.fetch(CHUNKS_SQL, (ctx.corpus_id,)), start=1):
        emit(_insert("chunks", {"rid": rid, **r}))
        # Full-text search over title and text, as the build database's `fts` column.
        emit(_insert("chunks_fts", {"rowid": rid, "body": f"{r['title'] or ''} {r['text']}"}))
    for r in await ctx.db.fetch(EDGES_SQL, (ctx.corpus_id,)):
        emit(_insert("edges", r))
    for r in await ctx.db.fetch(COMMUNITIES_SQL, (ctx.corpus_id,)):
        emit(_insert("communities", {"community_id": r["community_id"], "summary": r["summary"]}))
        for member in r["member_ids"] or []:
            emit(_insert("community_members", {"community_id": r["community_id"], "chunk_id": member}))
    flush()
    return names


async def _export_vectors(ctx: Context, out: Path, serving: dict[str, Any], per_file: int, done: set[str]) -> list[str]:
    """NDJSON vector files ({id, values}), the chunks embedded as the embed stage renders them; a file already
    written is not embedded again."""
    tpl = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False).from_string(ctx.reg.policy("embed.templates.chunk"))
    rows = await ctx.db.fetch(CHUNKS_SQL, (ctx.corpus_id,))
    names = [f"vectors-{i // per_file:04d}.ndjson" for i in range(0, len(rows), per_file)]
    embedder = None
    for i, name in enumerate(names):
        if name in done or (out / name).exists():
            continue
        embedder = embedder or build_embedder(serving, ctx.settings.device)
        part = rows[i * per_file:(i + 1) * per_file]
        texts = [tpl.render(heading_path=r["heading_path"] or [], text=r["text"], title=r["title"] or "") for r in part]
        vectors = await embedder.embed(texts)
        tmp = out / (name + ".tmp")
        tmp.write_text("".join(json.dumps({"id": r["chunk_id"], "values": [round(x, 6) for x in v]}) + "\n"
                               for r, v in zip(part, vectors, strict=True)))
        tmp.replace(out / name)
        log.get().info("cloudflare_vectors_written", file=name, vectors=len(part))
    return names


def _config(name: str, d1: dict[str, str], index: str, kv_id: str, allowed: str, s1: dict[str, Any], workers_ai: bool) -> dict:
    return {
        "name": name, "main": "src/index.ts", "compatibility_date": "2026-10-01", "compatibility_flags": ["nodejs_compat"],
        "d1_databases": [{"binding": "DB", "database_name": d1["name"], "database_id": d1["id"]}],
        "vectorize": [{"binding": "VECTORS", "index_name": index}],
        "kv_namespaces": [{"binding": OAUTH_KV, "id": kv_id}],
        "ai": {"binding": "AI"},
        "vars": {"ALLOWED_GITHUB_USERS": allowed, "SYSTEM_ONE_BINDING": "workers_ai" if workers_ai else ""},
        "observability": {"enabled": True},
    }


async def publish(ctx: Context, name: str, allowed_users: str | None) -> dict[str, Any]:
    pol = ctx.reg.policy("publish.cloudflare")
    serving = ctx.reg.profile("embedding", pol["serving_embedding"])
    allowed = allowed_users or os.environ.get(ALLOWED_USERS_ENV)
    if not allowed:
        raise CloudflareError(f"give the GitHub users allowed to sign in (--allow-github-users or ${ALLOWED_USERS_ENV})")
    root = (ctx.settings.workspace_root.parent / "cloudflare" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    top = _State(root / STATE_FILE)
    work = _worker_dir(root)
    wrangler = Wrangler(work)

    version = (await _fingerprint(ctx, serving))[:10]
    vdir = root / version
    vdir.mkdir(exist_ok=True)
    st = _State(vdir / STATE_FILE)
    resource = f"{name}-{version}"

    if not st["sql_files"]:
        st.set("sql_files", await _export_sql(ctx, vdir, pol["sql_file_bytes"]))
    if not st["d1"]:
        found = {d["name"]: d["uuid"] for d in wrangler.json("d1", "list")}
        if resource not in found:
            wrangler("d1", "create", resource)
            found = {d["name"]: d["uuid"] for d in wrangler.json("d1", "list")}
        st.set("d1", {"name": resource, "id": found[resource]})
    for f in [SCHEMA_FILE, *st["sql_files"]]:
        if f not in (st["imported"] or []):
            wrangler("d1", "execute", resource, "--remote", "--yes", f"--file={vdir / f}")
            st.add("imported", f)
            log.get().info("cloudflare_d1_imported", file=f)

    vector_files = await _export_vectors(ctx, vdir, serving, pol["vectors_per_file"], set(st["upserted"] or []))
    if not st["index"]:
        if resource not in {i["name"] for i in wrangler.json("vectorize", "list")}:
            wrangler("vectorize", "create", resource, f"--dimensions={serving['dim']}", "--metric=cosine")
        st.set("index", resource)
    for f in vector_files:
        if f not in (st["upserted"] or []):
            wrangler("vectorize", "upsert", resource, f"--file={vdir / f}", f"--batch-size={pol['vector_batch']}")
            st.add("upserted", f)

    if not top["kv"]:
        kv_title = f"{name}-oauth"
        found = {n["title"]: n["id"] for n in json.loads(wrangler("kv", "namespace", "list"))}
        if kv_title not in found:
            wrangler("kv", "namespace", "create", kv_title)
            found = {n["title"]: n["id"] for n in json.loads(wrangler("kv", "namespace", "list"))}
        top.set("kv", found[kv_title])

    s1 = bundle.system_one(ctx.jev.provider.profile)
    workers_ai = WORKERS_AI_RUN.match(s1["url"])
    (work / "src" / "bundle.json").write_text(json.dumps(bundle.build(ctx, serving), ensure_ascii=False, indent=1))
    (work / "wrangler.json").write_text(json.dumps(_config(name, st["d1"], st["index"], top["kv"], allowed, s1, bool(workers_ai)), indent=2))
    out = wrangler("deploy")
    url = next(iter(WORKERS_DEV_URL.findall(out)), top["url"])
    if top["current"] != version:
        top.set("previous", top["current"])
        top.set("current", version)
    top.set("url", url)

    have = {s["name"] for s in json.loads(wrangler("secret", "list"))}
    needed = [*SECRETS, *([] if workers_ai else [SYSTEM_ONE_KEY])]
    return {"url": url, "mcp_url": f"{url}/mcp" if url else None, "version": version, "previous": top["previous"],
            "d1": st["d1"]["name"], "vectorize": st["index"], "missing_secrets": [s for s in needed if s not in have],
            "github_callback_url": f"{url}/callback" if url else None}
