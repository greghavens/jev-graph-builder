---
name: continue_job
version: 1
purpose: "Resume message for a batch job that stopped before writing every record."
inputs: [missing_ids, output_file]
output_schema: null
enters: none
---
The previous run stopped before it finished. These record ids still have no valid line in
`{{ output_file }}`: {{ missing_ids | join(", ") }}.

Append one valid line for each of them, following the original instructions. Do not rewrite lines
that are already there.
