-- Chunk information density is no longer asked: nothing read it.
ALTER TABLE chunks DROP COLUMN IF EXISTS density;
