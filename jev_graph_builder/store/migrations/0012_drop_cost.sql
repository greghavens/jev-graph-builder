-- Remove the dollar columns from databases created before they were dropped.
ALTER TABLE jev_calls DROP COLUMN IF EXISTS cost_usd;
ALTER TABLE harness_runs DROP COLUMN IF EXISTS cost_usd;
