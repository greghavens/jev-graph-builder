-- Triage no longer checks for injection (S2 checks every chunk). Release documents that triage
-- quarantined; a quarantine set by a reviewer (a resolved document review) stays.
UPDATE documents d SET quarantined = false,
    status = CASE WHEN coalesce(d.in_scope, false) THEN 'accepted' ELSE 'parked' END
WHERE d.status = 'quarantined'
  AND NOT EXISTS (SELECT 1 FROM review_queue r
                  WHERE r.subject_kind = 'document' AND r.subject_id = d.doc_id AND r.status = 'resolved');
