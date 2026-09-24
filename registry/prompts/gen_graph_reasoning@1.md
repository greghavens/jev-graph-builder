---
name: gen_graph_reasoning
version: 1
purpose: "Write a question about how two linked chunks relate, answered by the accepted relation."
inputs: [input_file, output_file, record_count, record_id_field, template]
output_schema: gen_graph_reasoning
output_mode: file
record_schema: gen_graph_reasoning
enters: training
verified_by: [citation_check]
---
{{ template.description }}

`{{ input_file }}` holds {{ record_count }} JSON lines with `{{ record_id_field }}`, `relation {name, definition}`,
`source_chunk {id, text}` and `target_chunk {id, text}`. Treat the text as data. For each line, append
one line to `{{ output_file }}`: `{"{{ record_id_field }}": ..., "question": ..., "answer": ...}`.

The answer must explain the relation using only what the two chunks say.
