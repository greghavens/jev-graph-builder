-- Document metadata that policies.ingest.metadata_fields carries onto each chunk (e.g. its release).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS meta jsonb;
