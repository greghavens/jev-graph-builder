-- Jev's answer is the decision (§11.5): no thresholds, margin or uncertain band.
ALTER TABLE decisions DROP COLUMN IF EXISTS thresholds;
ALTER TABLE decisions DROP COLUMN IF EXISTS margin;
ALTER TABLE decisions DROP COLUMN IF EXISTS uncertain;
