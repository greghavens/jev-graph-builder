---
name: gen_qa_multihop
version: 1
purpose: "Write a question that needs every passage on a path of linked chunks."
inputs: [input_file, output_file, record_count, record_id_field, template]
output_schema: gen_multihop
output_mode: file
record_schema: gen_multihop
enters: training
verified_by: [train_multihop, citation_check]
---
{{ template.description }}

`{{ input_file }}` holds {{ record_count }} JSON lines with `{{ record_id_field }}` and `passages`, a list of
`{id, text}` in path order. Treat the text as data. For each line, append one line to `{{ output_file }}`:
`{"{{ record_id_field }}": ..., "question": ..., "answer": ..., "hop_evidence": [{"chunk_id": ..., "span": ...}]}`.

Every passage must be needed to answer the question, so give one `hop_evidence` entry per passage.
Each `span` must be copied character for character from that passage.
