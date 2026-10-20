-- A chunk whose text another document already carries is reused, not built again: each further document records
-- where it occurs. The chunk row stays with the document that first carried it.
CREATE TABLE IF NOT EXISTS chunk_occurrences (
  doc_id text REFERENCES documents, ord int, chunk_id text REFERENCES chunks, created_run_id text,
  PRIMARY KEY (doc_id, ord));
CREATE INDEX IF NOT EXISTS chunk_occurrences_chunk ON chunk_occurrences (chunk_id);
CREATE INDEX IF NOT EXISTS chunks_text_md5 ON chunks (corpus_id, md5(text));
