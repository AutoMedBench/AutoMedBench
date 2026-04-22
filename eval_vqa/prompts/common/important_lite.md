## Important
- This is an INFERENCE-ONLY benchmark. Load the VLM weights named in S1 — do
  NOT train, fine-tune, or distill any model.
- Process ALL question IDs found in the data directory — a missing
  `answer.json` for ANY question drops `completion_rate`, directly lowers
  the S5 score, and if many are missing will sink the run to Rating F.
- Use GPU for inference (CUDA). Never set `map_location='cpu'` when a GPU is
  available; load once with `torch_dtype=torch.float16` (or bf16) and reuse.
- **Batch inference is mandatory**, not optional. S3 runs all 15 calibration
  samples in a single `execute_code` call; S4 runs all remaining samples in
  a single `execute_code` call. Per-sample `execute_code` is the #1 cause of
  `time_limit_s: 3600` timeouts.
- Budget target: 20 samples should finish in < 5 min of S4 wall time; 500
  samples should finish in < 60 min with batched forward passes. If you see
  S4 approaching 30 min for < 100 samples, your loop is wrong — investigate.
- The winning strategy is NOT speed per turn — it is correct decode
  (full `raw_model_output`, short normalized `predicted_answer`). An agent
  that spends 2 min getting S3 postprocess right will beat one that skips
  calibration and ships broken short-phrase extraction.
- Print progress so the log captures which sample is being processed.
- When every required `answer.json` is written and you have verified the
  workspace, stop issuing `tool_calls`. The runner interprets an empty
  `tool_calls` response as "agent is done".
