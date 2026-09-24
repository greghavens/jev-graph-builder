-- P3 (§1, §19 item 4): chunk summaries, titles and keywords are harness output;
-- the Jev decisions that verified them are recorded on the chunk.
ALTER TABLE chunks ADD COLUMN summary_decision_ids text[];
