-- Provenance needed by drift sampling (§8.8), calibration from stored calls
-- (§13) and reversible entity merges (§8.6).
ALTER TABLE jev_calls ADD COLUMN questions jsonb;
ALTER TABLE jev_calls ADD COLUMN keys jsonb;
ALTER TABLE jev_calls ADD COLUMN dynamic jsonb;

CREATE TABLE entity_merges (
  merge_id text PRIMARY KEY, corpus_id text, entity_id text, merged_into text, decision_ids text[],
  status text, registry_version text, created_run_id text, created_at timestamptz DEFAULT now());
CREATE INDEX entity_merges_entity ON entity_merges (entity_id);

ALTER TABLE communities ADD COLUMN registry_version text;
ALTER TABLE communities ADD COLUMN created_run_id text;
ALTER TABLE review_queue ADD COLUMN question_set text;
ALTER TABLE review_queue ADD COLUMN run_id text;
ALTER TABLE review_queue ADD COLUMN created_at timestamptz DEFAULT now();
ALTER TABLE training_examples ADD COLUMN created_run_id text;
CREATE INDEX edges_corpus ON edges (corpus_id, structural, status);
CREATE INDEX entities_corpus ON entities (corpus_id, status);
CREATE INDEX chunks_status ON chunks (corpus_id, status);

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jev_graph_builder_writer') THEN
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA %I TO jev_graph_builder_writer', current_schema());
    EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO jev_graph_builder_reader', current_schema());
  END IF;
EXCEPTION WHEN insufficient_privilege THEN
  RAISE NOTICE 'skipping grants: insufficient privilege';
END $$;

-- §8.5: every vector records the model that produced it.
ALTER TABLE entities ADD COLUMN embedding_model_id text;
ALTER TABLE claims ADD COLUMN embedding_model_id text;

-- §8.6: `surface` is the extracted form (stable input to resolution);
-- `canonical_name` is the resolved name Jev chose for the merged cluster.
ALTER TABLE entities ADD COLUMN surface text;
