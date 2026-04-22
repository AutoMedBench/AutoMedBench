Verify outputs and finalise the submission.

1. Confirm every expected question id has an `${WORKSPACE_DIR}/<question_id>/answer.json`.
   Missing files drop `completion_rate` and the S5 score directly.
2. Validate each `answer.json` against the schema (`question_id`,
   `predicted_label`, `predicted_answer`, `raw_model_output`, `model_name`,
   `runtime_s`). Malformed JSON or missing required fields → sample treated
   as parse failure.
3. Cross-check internal consistency: `predicted_label` (if set) must be
   uppercase and in `valid_labels`; `predicted_answer` must either be a
   real short answer or be an honest-skip pair (`predicted_answer=""` AND
   `raw_model_output=""`). Do NOT pad with `"unknown"` / `"N/A"` / `"error"`
   — the scorer treats these as placeholders and caps S4 at 0.2.
4. Verify `${WORKSPACE_DIR}/smoke_forward.json` still exists and has
   `success: true`, `wall_s >= 0.3`, and a non-empty real
   `raw_output_sample`. Missing / invalid smoke forces S2 to 0.
5. Verify `${WORKSPACE_DIR}/answer_postprocess.py` and
   `${WORKSPACE_DIR}/s3_calibration.json` are present (required S3 artefacts).
6. Once everything is in place, stop issuing `tool_calls`. The runner
   interprets an empty tool_calls response as "agent is done" and will run
   format_checker + scorer on your submission.
