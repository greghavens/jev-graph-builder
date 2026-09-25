-- Mentions, claims and edges drawn from any superseded chunk are superseded too, and so is each entity left with
-- no live mention. Segmentation used to retire re-cut chunks without this; 0017 covered superseded documents only.
CREATE TEMP TABLE stale_chunks AS SELECT chunk_id FROM chunks WHERE status = 'superseded';

UPDATE edges SET status = 'superseded'
WHERE status <> 'superseded' AND (src_id IN (SELECT chunk_id FROM stale_chunks) OR dst_id IN (SELECT chunk_id FROM stale_chunks));
UPDATE claims SET status = 'superseded' WHERE chunk_id IN (SELECT chunk_id FROM stale_chunks) AND status <> 'superseded';
UPDATE mentions SET status = 'superseded' WHERE chunk_id IN (SELECT chunk_id FROM stale_chunks) AND status <> 'superseded';
UPDATE entities e SET status = 'superseded'
WHERE e.status <> 'superseded'
  AND e.entity_id IN (SELECT m.entity_id FROM mentions m WHERE m.chunk_id IN (SELECT chunk_id FROM stale_chunks))
  AND NOT EXISTS (SELECT 1 FROM mentions m WHERE m.entity_id = e.entity_id AND m.status <> 'superseded');

DROP TABLE stale_chunks;
