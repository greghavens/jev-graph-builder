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
jev-graph-builder serve                   # HTTP API
```

Every command prints JSON on stdout. Progress and logs go to stderr.

## Tests

```bash
pip install -e '.[dev]'
pytest tests/unit tests/contract                                  # no services needed
JGB_TEST_DSN=postgresql://user@host/db pytest tests/integration   # needs PostgreSQL with pgvector
```
