-- Ontology evolution (§8.7.4) tests the `other` rate over the S6 link picks it has not yet examined.
ALTER TABLE edges ADD COLUMN IF NOT EXISTS evolution_seen boolean NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS edges_evolution_window ON edges (corpus_id, rel)
  WHERE NOT structural AND NOT evolution_seen AND src_kind = 'chunk' AND dst_kind = 'chunk';
