# jev-graph-builder

Turns a folder of documents into a searchable knowledge graph in PostgreSQL (pgvector). A coding agent (Claude Code or Codex) drafts the text: extractions, summaries, an ontology. Jev checks every piece before it goes into the graph.

What you get:

- **Graph RAG for your assistants.** An MCP server that Claude Code, Codex, OpenCode, claude.ai and ChatGPT can query. It finds passages by search, then follows the graph's verified links to the related passages a plain vector search misses, and returns them all with citations.
- **Training data.** Q&A, multi-hop Q&A and fact sets drawn from the graph, for teaching a model what the documents say.
- **A record of every decision.** Each passage, fact and link in the graph traces back to the Jev decision that accepted it.

## How it works

1. **Read** your documents and keep the ones that match the corpus scope.
2. **Split** each document into chunks along its headings, with document metadata.
3. **Extract** the entities and facts each chunk states, and summarise it.
4. **Embed** the chunks for vector search (all-MiniLM-L6-v2 by default).
5. **Merge** entities that are the same thing under different names.
6. **Link** chunks that relate to each other with typed relations drafted from your documents, for example one elaborates, depends on or supersedes another. Candidate links come from nearest-neighbour search across the whole corpus, so a new document is linked to the existing ones.
7. **Audit** the result.

At each step the coding agent or code proposes, and [Jev](https://typesafe.ai) (TypeSafe's decision model) decides: it answers a fixed yes/no or multiple-choice question with a calibrated probability, and only what passes goes into the graph. When a source document changes, ingest replaces the old version and everything derived from it, and the new version is linked again.

## Quickstart

```bash
pip install git+https://github.com/greghavens/jev-graph-builder

export TYPESAFE_API_KEY=...        # your TypeSafe key
jev-graph-builder --harness claude_code build ./my-docs
```

- `./my-docs` is your documents (see [Your documents](#your-documents)).
- `--harness` is your coding agent: `claude_code`, `codex`, or `auto` (see [Coding agent](#coding-agent)).
- With no database given, `build` starts a local pgvector container with podman or docker.
- The first run copies a starter Registry (ontology, prompts, Jev question sets, policies) into `./registry`; S0 then adapts it to your documents.

The run pauses once, early on: it shows the ontology it drafted from your documents (entity types, relation types and so on) and asks you to approve it. Answer no and it stops; running `build` again asks again. After you approve, it runs to the end. If it stops for any reason, run the same command again and it picks up where it left off.

## Requirements

- Python 3.11 or later
- A TypeSafe API key, in `TYPESAFE_API_KEY`
- A coding agent CLI, installed and logged in: [Claude Code](https://claude.com/claude-code) (`claude`) and/or [Codex](https://github.com/openai/codex) (`codex`)
- Either podman or docker (for the automatic database), or your own PostgreSQL 16 with the pgvector extension
- Optional: Tesseract, for scanned PDFs

## Your documents

Pass any mix of files, directories and glob patterns:

```bash
jev-graph-builder build ./docs
jev-graph-builder build ./handbook ./release-notes/2026 extra/faq.md
jev-graph-builder build './site/**/*.html'
```

- Directories are read recursively. Hidden files (names starting with `.`) are skipped.
- Supported formats: Markdown, HTML, Word (`.docx`), PDF, and plain text (`.txt`, `.rst`, other text files). HTML is also recognised by its content when the file name doesn't say.
- Markdown front matter becomes document metadata.
- A file that can't be read is marked failed and the rest of the run carries on. `status` lists failures, and re-running `build` retries them.

## Coding agent

| `--harness` | Uses |
|---|---|
| `claude_code` | Claude Code only |
| `codex` | Codex only |
| `auto` (default) | Both. Jev picks the model for each job, so both CLIs must be installed and logged in. |

The option goes before the command: `jev-graph-builder --harness codex build ./docs`.

## GPU

Embedding and the nearest-neighbour searches that find candidate entity merges and chunk links run on the GPU when PyTorch can see one. `--device` picks for the whole run:

| `--device` | Uses |
|---|---|
| `auto` (default) | The GPU if there is one, otherwise the CPU |
| `cpu` | The CPU only |
| `cuda`, `cuda:1`, ... | That GPU |

Like `--harness`, it goes before the command: `jev-graph-builder --device cpu build ./docs`.

## Options

```bash
jev-graph-builder build <docs...> [--db <postgres-url>] [--corpus <id>] [--stop-after <stage>]
```

| Option | Meaning |
|---|---|
| `--db` | Use your own PostgreSQL instead of the automatic container |
| `--corpus` | Name for this document set, so one database can hold several |
| `--stop-after` | Stop after a stage (`bootstrap`, `ingest`, `segment`, `enrich`, ...) to inspect it; run again without it to continue |

The same settings can go in a `jev-graph-builder.yaml` file in the working directory instead of on the command line:

```yaml
dsn: postgresql://user@localhost:5432/graph
harness: claude_code
device: auto
```

Keep the TypeSafe key in the environment, not in this file.

## After the build

```bash
jev-graph-builder status                  # progress, counts and failures per stage
jev-graph-builder search "how do retries work" -k 10
jev-graph-builder serve --port 8080       # HTTP API
```

Every command prints JSON on stdout. Progress and logs go to stderr.

## Use the graph in Claude Code, Codex or OpenCode

`jev-graph-builder mcp` is an MCP server with two tools:

- `graph_context`: takes a question and returns cited passages (`p1`, `p2`, ...). Each has its text, heading path and source file, found by search and then by following the graph's verified links. Every passage found is returned, with no token cap. `answerable` is true when at least one passage was found. The agent session writes the answer itself.
- `search`: ranked chunks for a query, without following links.

The server needs the directory you ran `build` in (its `jev-graph-builder.yaml` and `registry/`) and your TypeSafe key. In the examples, replace `/path/to/build-dir` with that directory and `/path/to/jev-graph-builder` with where you installed this tool.

Name the server after what the documents cover (`vcf-docs` below), not after this tool: the agent sees tools as `mcp__vcf-docs__graph_context`, and a descriptive name helps it pick them for the right questions. Use that name as the client's server key and pass it with `--name` (or set `mcp_name` in `jev-graph-builder.yaml`) so the server reports the same name. The server also tells the session what the documents cover, from `corpus.yaml`'s `scope`.

With Claude Code you can register it for every session instead of one project:

```bash
claude mcp add --scope user vcf-docs \
  -e JEV_GRAPH_BUILDER_CONFIG=/path/to/build-dir/jev-graph-builder.yaml \
  -e JGB_REGISTRY_PATH=/path/to/build-dir/registry \
  -- /path/to/jev-graph-builder/.venv/bin/jev-graph-builder mcp --name vcf-docs
```

**Claude Code**: `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "vcf-docs": {
      "type": "stdio",
      "command": "/path/to/jev-graph-builder/.venv/bin/jev-graph-builder",
      "args": ["mcp", "--name", "vcf-docs"],
      "env": {
        "JEV_GRAPH_BUILDER_CONFIG": "/path/to/build-dir/jev-graph-builder.yaml",
        "JGB_REGISTRY_PATH": "/path/to/build-dir/registry",
        "TYPESAFE_API_KEY": "${TYPESAFE_API_KEY}"
      }
    }
  }
}
```

**Codex**: `~/.codex/config.toml`:

```toml
[mcp_servers.vcf-docs]
command = "/path/to/jev-graph-builder/.venv/bin/jev-graph-builder"
args = ["mcp", "--name", "vcf-docs"]
env = { JEV_GRAPH_BUILDER_CONFIG = "/path/to/build-dir/jev-graph-builder.yaml", JGB_REGISTRY_PATH = "/path/to/build-dir/registry" }
env_vars = ["TYPESAFE_API_KEY"]
default_tools_approval_mode = "approve"   # otherwise `codex exec` refuses the calls
tool_timeout_sec = 300
```

**OpenCode**: `opencode.json` in your project:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "vcf-docs": {
      "type": "local",
      "command": ["/path/to/jev-graph-builder/.venv/bin/jev-graph-builder", "mcp", "--name", "vcf-docs"],
      "enabled": true,
      "environment": {
        "JEV_GRAPH_BUILDER_CONFIG": "/path/to/build-dir/jev-graph-builder.yaml",
        "JGB_REGISTRY_PATH": "/path/to/build-dir/registry",
        "TYPESAFE_API_KEY": "{env:TYPESAFE_API_KEY}"
      }
    }
  }
}
```

To share one server between sessions, run `jev-graph-builder mcp --name vcf-docs --transport streamable-http --port 8765` and point clients at `http://127.0.0.1:8765/mcp`. Without `MCP_PUBLIC_URL` set, the HTTP server has no authentication, so keep it on `127.0.0.1` (the default). To run the server with least privilege, set `JGB_READER_DSN` to a database user with the `jev_graph_builder_reader` role. That role can read the graph and can only add records of its own Jev calls and decisions.

## Host it for claude.ai and ChatGPT

Web clients need the server on a public HTTPS address with OAuth sign-in. When `MCP_PUBLIC_URL` is set, `mcp --transport streamable-http` acts as its own OAuth server: clients register themselves, you sign in with GitHub, and only the GitHub logins in `MCP_ALLOWED_GITHUB_USERS` get a token. Sign-ins are kept in the database, so a restart doesn't sign anyone out.

| Environment variable | Value |
|---|---|
| `MCP_PUBLIC_URL` | The server's public address, e.g. `https://my-docs-mcp.fly.dev` |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | From a GitHub OAuth app (below) |
| `MCP_ALLOWED_GITHUB_USERS` | Comma-separated GitHub logins allowed to connect |
| `JGB_DSN` | The database holding the graph |
| `TYPESAFE_API_KEY` | Your TypeSafe key |

`deploy/fly/` has a working setup for [Fly.io](https://fly.io): a private Postgres + pgvector app (`db/`) and the MCP server (`mcp/`), which stops when idle and starts on the first request. Change the app names in both `fly.toml` files and `MCP_PUBLIC_URL` to your own, then:

1. **Database.** Create the app and its volume, set `POSTGRES_PASSWORD`, and deploy:
   ```bash
   fly apps create my-docs-db
   fly volumes create pgdata -a my-docs-db --size 20
   fly secrets set -a my-docs-db POSTGRES_PASSWORD=...
   fly deploy -c deploy/fly/db/fly.toml
   ```
2. **Load the graph.** With `fly proxy 15432:5432 -a my-docs-db` running, copy your build database into it with `pg_dump` and `psql` (`--no-owner --no-privileges`).
3. **GitHub OAuth app.** At [github.com/settings/applications/new](https://github.com/settings/applications/new), set the homepage to `MCP_PUBLIC_URL` and the callback URL to `MCP_PUBLIC_URL` + `/github/callback`. Leave wildcard redirects and device flow off. Generate a client secret.
4. **Server.** Copy your build directory's `registry/` to `deploy/fly/mcp/registry/` (it is git-ignored), set the secrets and deploy:
   ```bash
   fly apps create my-docs-mcp
   fly secrets set -a my-docs-mcp TYPESAFE_API_KEY=... GITHUB_CLIENT_ID=... GITHUB_CLIENT_SECRET=... \
     MCP_ALLOWED_GITHUB_USERS=your-login JGB_DSN=postgresql://postgres:...@my-docs-db.internal:5432/jev_graph_builder
   fly deploy . -c deploy/fly/mcp/fly.toml --dockerfile deploy/fly/mcp/Dockerfile
   ```
   Run `fly secrets set` in your own terminal, not through an AI agent, so the secrets stay out of its transcript.
5. **Connect.**
   - **claude.ai:** Settings → Connectors → Add custom connector, URL `MCP_PUBLIC_URL` + `/mcp`. Leave the OAuth fields blank.
   - **ChatGPT:** turn on Developer mode, then add a connector with the same URL and OAuth.

   Both send you to GitHub to sign in.

## Training data

After the build, `train` turns the knowledge in the graph into training data for a model that has to learn it. Each example records the Jev decisions that accepted it.

```bash
jev-graph-builder train build                          # templates from corpus.yaml training_templates
jev-graph-builder train build --sample 0.01           # a deterministic 1% sample first
jev-graph-builder train export --out training-export --format jsonl
```

| Template | What it holds |
|---|---|
| `qa_single` | One question per verified fact (claim), written by the coding agent and answered from the fact's chunk. At most `training.coverage.facts_per_chunk` per chunk; a chunk with no claims gets one question if it has `min_chunk_tokens`. |
| `qa_multihop` | A question that needs two or more linked chunks |
| `facts` | Every verified claim as a plain fact with its source chunk, for continued pretraining |

These three are the defaults. `retrieval_pairs`, `triples` (links between chunks), `rerank_soft_labels`, `instruction` and `graph_reasoning` are also available; the first three train search or graph models rather than teach the corpus.

Jev checks that the answer says nothing its source does not (`train_qa`, `citation_check`) and that the question states or assumes nothing about the documented products its source does not (`train_premise`). It also checks every question and answer against `registry/output_rules/` (no references to "the passage" or "this chunk", no answer leaked in the question, no trivia such as dates or file names). When a rule fails, the coding agent rewrites the example and Jev checks it again, up to `training.output_rules.reauthor_rounds` times. An example still in doubt goes to review.

Examples are split into train, validation and test by document, so no split contains another split's documents. An example whose documents fall in different splits is not exported. `--limit` is shared evenly across the templates, `--resume` finishes only what an earlier run left unfinished, and `--dry-run` shows what would run. `--format` takes `jsonl`, `parquet` or `hf`.

Q&A templates make one coding-agent call per example (about 164,000 for a corpus of 85,000 chunks), so try `--sample` before a full run.

## Tests

```bash
pip install -e '.[dev]'
pytest tests/unit tests/contract                                  # no services needed
JGB_TEST_DSN=postgresql://user@host/db pytest tests/integration   # needs PostgreSQL with pgvector
```

## License

[Apache License 2.0](LICENSE).
