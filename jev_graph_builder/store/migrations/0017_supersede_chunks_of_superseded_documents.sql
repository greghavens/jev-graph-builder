-- Chunks of a superseded document version are superseded too, with what was drawn from them: edges that touch
-- them, their mentions and claims, and entities left with no live mention. Ingest did not do this before.
CREATE TEMP TABLE stale_chunks AS
SELECT c.chunk_id FROM chunks c JOIN documents d USING (doc_id) WHERE d.status = 'superseded';

UPDATE chunks SET status = 'superseded' WHERE chunk_id IN (SELECT chunk_id FROM stale_chunks) AND status <> 'superseded';
UPDATE edges SET status = 'superseded'
WHERE status <> 'superseded' AND (src_id IN (SELECT chunk_id FROM stale_chunks) OR dst_id IN (SELECT chunk_id FROM stale_chunks));
UPDATE claims SET status = 'superseded' WHERE chunk_id IN (SELECT chunk_id FROM stale_chunks) AND status <> 'superseded';
UPDATE mentions SET status = 'superseded' WHERE chunk_id IN (SELECT chunk_id FROM stale_chunks) AND status <> 'superseded';
UPDATE entities e SET status = 'superseded'
WHERE e.status <> 'superseded'
  AND e.entity_id IN (SELECT m.entity_id FROM mentions m WHERE m.chunk_id IN (SELECT chunk_id FROM stale_chunks))
  AND NOT EXISTS (SELECT 1 FROM mentions m WHERE m.entity_id = e.entity_id AND m.status <> 'superseded');

DROP TABLE stale_chunks;
