## Important
- This is an INFERENCE-ONLY benchmark. Load pre-trained VLM weights — do
  NOT train, fine-tune, or distill any model.
- Process ALL question IDs found in the data directory — a missing
  `answer.json` for ANY question drops `completion_rate`, directly lowers
  the S5 score, and if many are missing will sink the run to Rating F.
- Use GPU for inference (CUDA). Load each candidate VLM once per process
  with `torch_dtype=torch.float16` (or bf16) and reuse across samples.
- **Batch inference is mandatory**, not optional. S3 runs all 15 calibration
  samples in a single `execute_code` call; S4 runs all remaining samples in
  a single `execute_code` call per chosen model. Per-sample `execute_code`
  is the #1 cause of `time_limit_s: 3600` timeouts.
- Budget target: 20 samples should finish in < 5 min of S4 wall time per
  model; 500 samples should finish in < 60 min with batched forward passes.
- The winning strategy is NOT speed of decision — it is picking the right
  VLM and getting decode + short-answer extraction correct. An agent that
  spends 5 min on candidate research and picks a mediocre VLM will lose to
  one that spends 15 min comparing 2–3 candidates head-to-head on a held-out
  calibration set.
- Print progress so the log captures which sample is being processed.
- When every required `answer.json` is written and you have verified the
  workspace, stop issuing `tool_calls`. The runner interprets an empty
  `tool_calls` response as "agent is done".
