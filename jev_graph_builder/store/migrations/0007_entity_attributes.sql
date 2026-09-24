-- §8.4 step 1: the harness extracts entity attributes; the Jev-verified ones are
-- kept on the entity row (merged entities keep the union, first value wins).
ALTER TABLE entities ADD COLUMN IF NOT EXISTS attributes jsonb;
