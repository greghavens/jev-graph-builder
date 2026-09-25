# jev-graph-builder

Turns a folder of documents into a searchable knowledge graph in PostgreSQL (pgvector). A coding agent (Claude Code or Codex) drafts the text: extractions, summaries, an ontology. Jev checks every piece before it goes into the graph.

## Quickstart

```bash
git clone https://github.com/greghavens/jev-graph-builder
cd jev-graph-builder
python -m venv .venv && .venv/bin/pip install -e .

export TYPESAFE_API_KEY=...        # your TypeSafe key
.venv/bin/jev-graph-builder --harness claude_code build ./my-docs
```

- `./my-docs` is your documents (see [Your documents](#your-documents)).
- `--harness` is your coding agent: `claude_code`, `codex`, or `auto` (see [Coding agent](#coding-agent)).
- With no database given, `build` starts a local pgvector container with podman or docker.

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

- `graph_context`: takes a question and returns cited passages (`p1`, `p2`, ...). Each has its text, heading path and source file, found by search and then by following the graph's verified links. Jev decides whether they can answer the question; if not, `answerable` is false and no passages come back. The agent session writes the answer itself.
- `search`: ranked chunks for a query, without following links.

The server needs the directory you ran `build` in (its `jev-graph-builder.yaml` and `registry/`) and your TypeSafe key. In the examples, replace `/path/to/build-dir` with that directory and `/path/to/jev-graph-builder` with where you installed this tool.

**Claude Code**: `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "jev-graph": {
      "type": "stdio",
      "command": "/path/to/jev-graph-builder/.venv/bin/jev-graph-builder",
      "args": ["mcp"],
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
[mcp_servers.jev-graph]
command = "/path/to/jev-graph-builder/.venv/bin/jev-graph-builder"
args = ["mcp"]
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
    "jev-graph": {
      "type": "local",
      "command": ["/path/to/jev-graph-builder/.venv/bin/jev-graph-builder", "mcp"],
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

To share one server between sessions, run `jev-graph-builder mcp --transport streamable-http --port 8765` and point clients at `http://127.0.0.1:8765/mcp`. The HTTP server has no authentication, so keep it on `127.0.0.1` (the default). To run the server with least privilege, set `JGB_READER_DSN` to a database user with the `jev_graph_builder_reader` role. That role can read the graph and can only add records of its own Jev calls and decisions.

## Tests

```bash
pip install -e '.[dev]'
pytest tests/unit tests/contract                                  # no services needed
JGB_TEST_DSN=postgresql://user@host/db pytest tests/integration   # needs PostgreSQL with pgvector
```
