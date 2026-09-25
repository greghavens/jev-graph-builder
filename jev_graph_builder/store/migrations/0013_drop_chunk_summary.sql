-- Chunk summaries are no longer produced: every consumer already has the chunk text.
ALTER TABLE chunks DROP COLUMN IF EXISTS summary;
