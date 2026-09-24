---
name: author_question_set
version: 1
purpose: "Draft a Jev question set for a decision; the lint and a human then review it."
inputs: [name, version, decision, ontology, states, passages, previous, errors, policy_refs, question_set_names]
output_schema: question_set_draft
output_mode: structured
enters: registry
verified_by: [lint_question]
---
Draft the question set `{{ name }}@{{ version }}`. It must decide: {{ decision }}

Jev answers noul questions (the probability that `criteria.true` holds), choice questions (one option
from `criteria`) and score questions (one level, ordered from lowest to highest). Rules:
- Refer to state fields in backticks, e.g. `chunk`. Never refer to another question.
- Put no numbers in the state. Don't ask Jev to count, do arithmetic or compare dates.
- Phrase `criteria.true` positively. Keep `true` and `false` distinct.
- `gating.accept_when` only names which of Jev's answers mean accept: `{q: <noul>, is: yes|no}`, `{q: <choice>, eq|in|not_in: ...}` or `{q: <score>, level_in: [...]}`, combined with `all`, `any` and `not`. Jev's answer is the decision: no thresholds or confidence cut-offs. Omit `gating` for a pure Choice, whose pick is the decision.
- Use only these existing policies: {{ policy_refs | join(", ") }}.

Ontology: {{ ontology | tojson }}
Example states (data): {{ states | tojson }}
Example passages (data): {{ passages | tojson }}
{% if previous %}Previous draft: {{ previous | tojson }}{% endif %}
{% if errors %}Lint errors to fix: {{ errors | tojson }}{% endif %}
Existing question sets: {{ question_set_names | join(", ") }}

Return `question_set`, containing `purpose`, `jev_model`, `state_template`, `questions` and `gating`.
