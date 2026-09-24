-- Operational extensions: Registry approvals, alerts, and lookup indexes for
-- the Jev cache (§11.6), the ledger (§14) and the audit reports (§16).
CREATE TABLE registry_versions (
  version text PRIMARY KEY, corpus_id text, status text, summary jsonb,
  approved_by text, approved_at timestamptz, activated_at timestamptz, created_at timestamptz DEFAULT now());

CREATE TABLE alerts (alert_id text PRIMARY KEY, kind text, subject text, payload jsonb, run_id text, created_at timestamptz DEFAULT now());

ALTER TABLE jev_calls ADD COLUMN provider text;
ALTER TABLE jev_calls ADD COLUMN cache_key text;
ALTER TABLE jev_calls ADD COLUMN drift boolean DEFAULT false;
CREATE INDEX jev_calls_cache ON jev_calls (cache_key) WHERE NOT drift;
CREATE INDEX jev_calls_run ON jev_calls (run_id);
ALTER TABLE decisions ADD COLUMN answers jsonb;
ALTER TABLE decisions ADD COLUMN created_at timestamptz DEFAULT now();
CREATE INDEX decisions_subject ON decisions (subject_kind, subject_id);
CREATE INDEX decisions_qs ON decisions (question_set, outcome);
CREATE INDEX work_items_status ON work_items (stage, status);
CREATE INDEX review_status ON review_queue (status, subject_kind);
CREATE INDEX mentions_entity ON mentions (entity_id);
CREATE INDEX mentions_chunk ON mentions (chunk_id);
CREATE INDEX claims_chunk ON claims (chunk_id);
CREATE INDEX claims_embedding_hnsw ON claims USING hnsw (embedding vector_cosine_ops) WITH (m = {{ hnsw_m }}, ef_construction = {{ hnsw_ef_construction }});
CREATE INDEX harness_runs_run ON harness_runs (run_id);
CREATE INDEX training_template ON training_examples (template, split);
ALTER TABLE harness_runs ADD COLUMN job_key text;
CREATE INDEX harness_runs_job ON harness_runs (job_key);
