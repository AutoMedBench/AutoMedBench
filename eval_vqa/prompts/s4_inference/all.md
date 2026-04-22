S4  INFERENCE   Run the requested evaluation slice and save exactly one
`answer.json` record per question. Do not silently skip failures.

**GPU required — same as S3.** You MUST use GPU (CUDA) for inference.
Ensure model and data are on GPU. Never use `map_location='cpu'` when a
GPU is available. Use `torch_dtype=torch.float16` (or bf16) and
`torch.inference_mode()` for throughput.

**Single-pass batch inference — MANDATORY.** S4 **MUST** run all remaining
samples in a **single `execute_code` call** that iterates the full
`question_ids` list inside one long-lived Python process. Do NOT:
- issue one `execute_code` per sample (each call re-pays Python startup,
  imports, and message overhead — this is the #1 reason runs hit
  `time_limit_s: 3600` before finishing);
- split the list into many small `execute_code` blocks "for safety";
- re-enter the agent loop between samples to "check progress".
One execute_code call, one for-loop over ALL samples, one model load —
that is the only acceptable shape. Budget target: 20 samples < 5 min,
500 samples < 60 min with batched forward passes (batch size 4–8 fits in
24 GB for LLaVA-Med 7B at fp16).

**Load the model once.** The VLM weights (`~7B` for LLaVA-Med) must be loaded
**exactly once** in a single long-lived Python process that then iterates over
all question IDs with a plain `for qid in question_ids:` loop. Do **not**:
- spawn a new Python subprocess per sample (e.g. `subprocess.run([...])` in
  a loop that each time re-imports transformers and calls `from_pretrained`);
- call `load_pretrained_model(...)` / `.from_pretrained(...)` inside the
  per-sample loop body;
- reload weights between batches unless you explicitly swap models.

Per-sample reload turns a ~15 min benchmark into a multi-hour run and will
cause time_limit_s to trip. If you must split the work across GPUs, load once
per GPU process, not once per sample. Prefer `torch.inference_mode()` and
`torch_dtype=torch.float16` (or bf16) for throughput; batching is optional but
single-load is mandatory.

**Real inference required.** Each `raw_model_output` must be the actual
decoded text from your local VLM's `.generate(...)` (or equivalent) call on
the real image+question pair. Do not substitute heuristic strings, keyword
rules, hardcoded answers, or placeholders (e.g. `heuristic:...`,
`fallback:...`, `unknown`, empty strings) when the model errors — investigate
and fix the error instead.

**You must NOT answer from your own knowledge.** You (the coding agent) are
*not* the VLM being benchmarked. You may not look at the image/question and
write the answer yourself, even if you are confident you know the correct
label. Every answer must come from the external VLM you loaded in S2. If
the external VLM errors on a particular sample, call `submit_answer` with
**both** `predicted_answer=""` AND `raw_model_output=""` — the scorer treats
matching empties as an honest skip and drops the sample from `valid_outputs`
without penalizing the run. Never pad with `"unknown"` / `"error"` / `"N/A"`,
and never leave a half-empty record (answer filled, raw empty) — both trip
`placeholder_rate` and cap S4 at 0.2. A run with honest empties beats a run
with fabricated answers: the scorer flags the latter as `E5 NO_MODEL_CALL`
and rates it F.

The scorer enforces this at two levels:
- Per sample: empty or placeholder `raw_model_output` is excluded from
  `parse_rate` / `valid_outputs`.
- Per run: if `placeholder_rate > 0.05` the scorer caps S4 at 0.2; if no
  evidence of a `.generate()` call is found in the conversation trace the
  scorer caps S4 at 0.3 and marks the run as `E5_NO_MODEL_CALL`.

**Use the S3 post-processor.** The S3 calibration stage must have produced
`${WORKSPACE_DIR}/answer_postprocess.py`. S4 inference code must
`from answer_postprocess import postprocess` and set
`predicted_answer = postprocess(raw_model_output)` for every sample before
calling `submit_answer(...)`. Do not re-implement ad-hoc normalization in S4
— iterate on `answer_postprocess.py` during S3 instead.

**Answer format (open-ended tasks).** For tasks with
`answer_mode: open_ended` (PathVQA, VQA-RAD), `predicted_answer` must be a
**normalized short answer**, not a full sentence. The scorer computes
`0.5 * EM + 0.5 * token_F1` after lowercasing / stripping punctuation;
writing a long sentence drives both to zero.

Rules for `predicted_answer`:
- Yes/no questions (gold is `yes` or `no`): write exactly `yes` or `no`,
  nothing else. Do **not** write `yes, the cyst wall shows ...`.
- Open short answers (one word or a short noun phrase): write the phrase
  only — no leading articles (`the`/`a`/`an`), no trailing period, no
  explanation. Example: `microscopy`, `region of epiphysis`, `benign`,
  `adenocarcinoma`.
- `raw_model_output` may be the model's full decoded text (sentence,
  reasoning, etc.) — you only need to extract and normalize the answer
  span into `predicted_answer`.

Good vs. bad examples:

```json
// Gold "no" — good
{"raw_model_output": "No, the upper dermis shows only mild edema.",
 "predicted_answer": "no"}

// Gold "no" — bad (EM=0, F1≈0.1)
{"raw_model_output": "No, the upper dermis shows only mild edema.",
 "predicted_answer": "no, the upper dermis shows only mild edema."}

// Gold "microscopy" — good
{"raw_model_output": "The image shows a microscopy of liver tissue.",
 "predicted_answer": "microscopy"}

// Gold "microscopy" — bad
{"raw_model_output": "The image shows a microscopy of liver tissue.",
 "predicted_answer": "the image shows a histopathological view of a"}
```

For `answer_mode: multiple_choice` tasks, `predicted_answer` is the option
text for `predicted_label` and `predicted_label` must be one of `A-E`; the
open-ended rules do not apply.

**After all samples are done, confirm that every question id has a
corresponding `${WORKSPACE_DIR}/<question_id>/answer.json`.** Print the
list of missing IDs (if any) to the log so you can see completion at a
glance. Do NOT proceed to S5 until the answer-file count matches the
resolved question count from the preamble.
