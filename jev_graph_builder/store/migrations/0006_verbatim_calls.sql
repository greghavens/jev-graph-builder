-- §8.8 drift sampling re-asks a stored call verbatim. `jsonb` normalises object
-- key order, which would reorder a choice question's options and the state's
-- fields on replay; `json` keeps the text exactly as it was sent.
ALTER TABLE jev_calls
  ALTER COLUMN state TYPE json USING state::json,
  ALTER COLUMN questions TYPE json USING questions::json,
  ALTER COLUMN dynamic TYPE json USING dynamic::json;
