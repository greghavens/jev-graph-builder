-- Least-privilege roles (§17). Skipped silently when the migrating user
-- cannot create roles (managed Postgres); an operator then applies it.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jev_graph_builder_writer') THEN
    CREATE ROLE jev_graph_builder_writer NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jev_graph_builder_reader') THEN
    CREATE ROLE jev_graph_builder_reader NOLOGIN;
  END IF;
  EXECUTE format('GRANT USAGE ON SCHEMA %I TO jev_graph_builder_writer, jev_graph_builder_reader', current_schema());
  EXECUTE format('GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA %I TO jev_graph_builder_writer', current_schema());
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO jev_graph_builder_reader', current_schema());
  -- the query API writes its own audit rows and review items
  GRANT INSERT ON jev_calls, decisions, harness_runs, review_queue TO jev_graph_builder_reader;
EXCEPTION WHEN insufficient_privilege THEN
  RAISE NOTICE 'skipping role creation: insufficient privilege';
END $$;
