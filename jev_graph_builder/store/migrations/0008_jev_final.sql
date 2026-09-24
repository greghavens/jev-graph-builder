-- §11.6: Jev is the last model to decide. There is no harness escalation ladder to
-- record; an uncertain answer takes its question set's safe default and is flagged.
ALTER TABLE decisions DROP COLUMN IF EXISTS escalation_step;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS uncertain boolean NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS decisions_uncertain_qs ON decisions (question_set, run_id) WHERE uncertain;
