S2  SETUP       Set up your environment and download the chosen VLM(s).
  1. Create a virtual environment for this run under
     `${WORKSPACE_DIR}/env/` using `python -m venv`. Activate it and install
     any additional packages your chosen VLM requires. Use this venv for
     all subsequent steps.
  2. Download model weights, processor / tokenizer assets, and any required
     dependencies into `${WORKSPACE_DIR}/`.
  3. Load the model into GPU memory and confirm it initializes without
     errors.
  4. Check the model's expected input format (image preprocessing,
     conversation template, image-token handling) and verify compatibility
     with the staged sample data before running inference at scale.
  5. Run a single smoke forward pass on one staged sample and verify the
     decoded text is a real non-empty English string before continuing.

**You are NOT the VLM.** Load a separate external VLM from HuggingFace (the
model you selected in S1) and run it on GPU. Answering questions from your
own multimodal reasoning without loading this external model is a
disqualifying violation — the scorer will tag the run `E5 NO_MODEL_CALL`
and cap S4 at 0.3.

**Required artefact — `${WORKSPACE_DIR}/smoke_forward.json`.** After your
smoke forward pass, write this file with the following schema:

```json
{
  "model_name": "<HF repo id you actually loaded>",
  "device": "cuda" | "cuda:0" | "...",
  "wall_s": <float, end-to-end seconds for the single forward pass, must be >= 0.3>,
  "raw_output_sample": "<first ~200 chars of the raw decoded model text>",
  "success": true
}
```

The scorer reads this artefact directly. If `smoke_forward.json` is missing,
`success` is not `true`, `wall_s < 0.3`, or `raw_output_sample` is empty /
starts with `heuristic:` / `fallback:` / `placeholder:`, **S2 will be forced
to 0.0 regardless of any other evidence**. Do not fabricate the file — the
`raw_output_sample` must come from a real `model.generate(...)` (or
equivalent) call on an actual image in the task.

**`raw_output_sample` must be a non-empty string.** If decode comes back
empty, do not write `success: true`; instead debug the generation call
(prompt formatting, `input_ids` slicing, EOS handling) until a real
non-empty decode is produced. An empty decode almost always means the model
saw a malformed prompt and emitted EOS immediately.

If a candidate VLM persistently returns `""` from `.generate(...)` under its
documented pipeline, mark it unusable in `plan.md` and fall back to the
next candidate rather than shipping a broken smoke.
