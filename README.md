# jev-graph-builder

Builds a verified, vector-searchable knowledge graph (PostgreSQL + pgvector) from a document corpus.

- Jev System One makes every decision: chunk boundaries, what is extracted, which entities are the same, which links exist. No other model decides anything (P0).
- The coding-agent harness (Claude Code or Codex) only generates text Jev cannot: the Registry draft, extractions, summaries. Nothing a harness writes enters the graph without an accepting Jev decision (P3).
- All domain knowledge lives in the versioned Registry (`registry/`), never in code (P1).
- Question answering and training-data generation are specified separately in `jev-graph-builder-qa-spec.md`.

The design is specified in `jev-graph-builder-spec.md`. Section numbers (§) below refer to it.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
export TYPESAFE_API_KEY=...          # never put secrets in the config file or the Registry
```

- PostgreSQL 16 with pgvector. Without `--db`, `build` starts and reuses its own local container (`profiles.postgres`, podman or docker).
- Migrations are generated from the active embedding profile. The vector dimension and HNSW parameters come from `registry/profiles.yaml`.
- Tesseract is optional. It is only used for scanned PDFs, per the `ocr` policy.

Configuration comes from `jev-graph-builder.yaml` (or the path in `$JEV_GRAPH_BUILDER_CONFIG`) and `JGB_*` environment variables. Nested keys use `__`, for example `JGB_PROFILES__JEV=typesafe`.

## Usage

```bash
jev-graph-builder --harness auto build <docs...> [--db <dsn>] [--corpus <id>] [--stop-after <stage>]
```

That one command does everything: S0 (the harness drafts the Registry, Jev verifies it), then S1..S7, then verification. You are asked for input only at one gate: the drafted Registry (once per corpus). Every other decision is Jev's.

Re-running the same command resumes where it stopped. `--stop-after bootstrap|ingest|segment|enrich|...` stops after that stage so each stage's output can be inspected before the next one runs; re-run without it (or with a later stage) to continue.

Documents may be Markdown, HTML (detected by content too), DOCX, PDF or text. HTML is converted to Markdown generically (`policies.ingest.html`); Markdown front matter is metadata, and S0 proposes which metadata keys (e.g. a release) Jev should see. Jev's answer is the decision: yes is yes, no is no, with no thresholds or uncertain band in code (§11.5). A chunk longer than the embedding window is split where Jev chooses.

Other commands (`bootstrap`, `ingest`, `run <stage>`, `plan`, `registry ...`, `search`, `serve`, `audit`, `status`) run single steps; `--harness claude_code|codex|auto` overrides the harness per command.

## Layout

| Path | Contents |
|---|---|
| `jev_graph_builder/registry/` | loader, JSON Schemas, lint, versioned proposals |
| `jev_graph_builder/jev/` | TypeSafe client and limiter, state / question builders, gating, service with cache and audit rows |
| `jev_graph_builder/harness/` | Claude Code / Codex adapters, batch jobs |
| `jev_graph_builder/pipeline/` | stages S0–S7 and the ledger-driven runner (S8 belongs to the QA spec) |
| `jev_graph_builder/query/` | hybrid search, HTTP API |
| `jev_graph_builder/calibrate/`, `audit/` | question-set drafting, drift; P3 / status reports |
| `jev_graph_builder/store/` | psycopg pool, templated migrations, upserts |
| `registry/` | seed Registry: corpus, ontology, question sets, prompts, schemas, policies, profiles, training templates |

## Tests

```bash
pytest tests/unit tests/contract                  # no services needed
JGB_TEST_DSN=postgresql://u:p@host/db pytest tests/integration   # or a container runtime for testcontainers
python -m jev_graph_builder.lint.hardcoding        # R-002 hard-coding lint (lint_allowlist.yaml)
```

- Contract tests replay a recorded live Jev response and harness event streams.
- Integration tests run the whole pipeline against a fixture corpus, with a scripted fake Jev and harness.
  - `test_resume.py` kills `run --all` at random points (5 trials) and checks that the resumed state is byte-identical to an uninterrupted run.
  - `JGB_TEST_KEEP_DB=1` keeps the test databases for inspection.
