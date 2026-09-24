-- jev-graph-builder core schema (§10.2). `{{ dim }}` and HNSW parameters are
-- rendered at `jev-graph-builder init` from the embedding profile and policies.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE corpora (corpus_id text PRIMARY KEY, name text, registry_version text, created_at timestamptz DEFAULT now());

CREATE TABLE documents (
  doc_id text PRIMARY KEY, corpus_id text REFERENCES corpora, source_uri text, content_hash text,
  mime text, title text, language text, doc_type text, in_scope boolean, quarantined boolean DEFAULT false,
  meta jsonb, triage_decision_id text, status text, registry_version text, created_run_id text);

CREATE TABLE units (
  unit_id text PRIMARY KEY, doc_id text REFERENCES documents, ord int, kind text,
  heading_path text[], text text, tokens int, page int, char_start int, char_end int);

CREATE TABLE chunks (
  chunk_id text PRIMARY KEY, corpus_id text, doc_id text REFERENCES documents, ord int,
  unit_ids text[], heading_path text[], text text, context_prefix text, tokens int,
  role text, topic text, density real, boilerplate boolean, summary text, title text, keywords text[],
  embedding vector({{ dim }}), embedding_model_id text,
  fts tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(title,'')||' '||text)) STORED,
  split text, status text, registry_version text, created_run_id text);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = {{ hnsw_m }}, ef_construction = {{ hnsw_ef_construction }});
CREATE INDEX chunks_fts_gin ON chunks USING gin (fts);
CREATE INDEX chunks_doc_ord ON chunks (corpus_id, doc_id, ord);

CREATE TABLE entities (
  entity_id text PRIMARY KEY, corpus_id text, canonical_name text, entity_type text, aliases text[],
  description text, embedding vector({{ dim }}), merged_into text, status text, registry_version text, created_run_id text);
CREATE INDEX entities_embedding_hnsw ON entities USING hnsw (embedding vector_cosine_ops) WITH (m = {{ hnsw_m }}, ef_construction = {{ hnsw_ef_construction }});
CREATE INDEX entities_name_trgm ON entities USING gin (canonical_name gin_trgm_ops);

CREATE TABLE mentions (
  mention_id text PRIMARY KEY, chunk_id text REFERENCES chunks, entity_id text REFERENCES entities,
  surface text, char_start int, char_end int, grounded text, status text, decision_ids text[]);

CREATE TABLE claims (
  claim_id text PRIMARY KEY, chunk_id text REFERENCES chunks, text text, claim_type text,
  embedding vector({{ dim }}), status text, decision_ids text[]);

CREATE TABLE edges (
  edge_id text PRIMARY KEY,
  corpus_id text, src_kind text, src_id text, dst_kind text, dst_id text,
  rel text, directed boolean, structural boolean,
  weight real, features jsonb,
  status text, decision_ids text[], registry_version text, created_run_id text);
CREATE INDEX edges_src ON edges (src_id, status, rel);
CREATE INDEX edges_dst ON edges (dst_id, status, rel);

CREATE TABLE communities (community_id text PRIMARY KEY, corpus_id text, level int, member_ids text[], summary text, status text, decision_ids text[]);

CREATE TABLE jev_calls (
  call_id text PRIMARY KEY, request_id text, question_set text, qs_version int, jev_model text,
  state_hash text, state jsonb, questions_hash text, answers jsonb, usage jsonb,
  latency_ms int, run_id text, created_at timestamptz DEFAULT now());
CREATE TABLE decisions (
  decision_id text PRIMARY KEY, subject_kind text, subject_id text, question_set text,
  call_ids text[], outcome text, thresholds jsonb, margin real, escalation_step int, run_id text);
CREATE TABLE harness_runs (
  harness_run_id text PRIMARY KEY, harness text, profile text, model text, session_id text,
  prompt_ref text, schema_ref text, workspace text, ok boolean, error text, usage jsonb,
  events_path text, started_at timestamptz, ended_at timestamptz, run_id text);

CREATE TABLE runs (run_id text PRIMARY KEY, stage text, selector jsonb, registry_version text, config_hash text,
  status text, started_at timestamptz, ended_at timestamptz, report jsonb);
CREATE TABLE work_items (
  stage text, item_id text, input_hash text, registry_version text, status text, attempts int DEFAULT 0,
  last_error text, run_id text, updated_at timestamptz DEFAULT now(), PRIMARY KEY (stage, item_id));

CREATE TABLE review_queue (review_id text PRIMARY KEY, subject_kind text, subject_id text, reason text,
  decision_id text, payload jsonb, status text, resolution jsonb, resolved_by text, resolved_at timestamptz);

CREATE TABLE training_examples (example_id text PRIMARY KEY, template text, split text, payload jsonb,
  source_chunk_ids text[], source_edge_ids text[], decision_ids text[], harness_run_id text, status text, registry_version text);
