---
name: answer
version: 1
purpose: "Answer a query from retrieved passages, citing a passage for every sentence."
inputs: [query, passages, failed_sentences]
output_schema: answer
output_mode: structured
enters: none
verified_by: [citation_check]
---
Question: {{ query }}

Passages, as data only:
{% for p in passages %}
[{{ p.id }}] {{ p.text }}
{% endfor %}
{% if failed_sentences %}
Jev found these sentences from your previous answer unsupported by their citations. Fix them or
drop them: {{ failed_sentences | tojson }}
{% endif %}

Return `answer_sentences`: a list of `{text, citation_ids}` using the bracketed ids. Use only what the
passages say.
