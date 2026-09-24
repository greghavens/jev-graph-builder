---
name: repair_output
version: 1
purpose: "Single repair turn after a structured answer failed schema validation."
inputs: [error]
output_schema: null
enters: none
---
Your last answer did not match the required output schema:

{{ error }}

Reply again with only a JSON object that matches the schema. Keep the content the same.
