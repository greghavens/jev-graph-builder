-- Reuse of stored answers across question set versions looks calls up by question set and state.
CREATE INDEX IF NOT EXISTS jev_calls_state ON jev_calls (question_set, state_hash) WHERE NOT drift;
