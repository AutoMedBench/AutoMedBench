S3  VALIDATE    Run inference on **at least 10 samples** (15 recommended)
from the public split as a calibration pass (not 1–3). Use this stage to
observe how the VLM's raw prose output maps to the task's short-phrase gold
answers and to design a deterministic post-processor before S4 starts at
scale.

**GPU required.** You MUST use GPU (CUDA) for inference. Verify with
`torch.cuda.is_available()`. Load the model onto GPU (e.g. `model.cuda()`
or `device='cuda'`). If `torch.load` uses `map_location`, set it to
`torch.device('cuda')`. Only fall back to CPU if CUDA is genuinely
unavailable — never force CPU when a GPU is present.

**Single-pass calibration — MANDATORY.** Run the calibration forward pass
in a **single `execute_code` call**: load the VLM once, loop over all 10+
ids in one Python process, write `s3_calibration.json` at the end. Do NOT
issue one `execute_code` per sample — that pattern exhausts
`time_limit_s` before S4 can start. Per-sample reload of the VLM in S3 is
the same anti-pattern the S4 prompt forbids.

**Debug budget.** If you have been stuck debugging the calibration pipeline
for more than **600 seconds** without a working non-empty decode, abandon
the current approach and go back to S1 to choose a different model, prompt
template, or decoding strategy. Do NOT burn the full `time_limit_s: 3600`
on a broken S3 — a run that hits S4 with a partial 10-sample calibration
beats a run that spends an hour stuck in S3 and writes no `answer.json`.

**Turn budget — commit, don't loop.** S3 calibration should take no more than
**10 `execute_code` turns**. If your postprocess/calibration approach isn't
working after 10 turns, commit to the best-so-far version, write
`s3_calibration.json` with whatever real VLM outputs you have (≥ 10
records), and proceed to S4 full inference. A partial calibration is
better than a stalled run with zero `answer.json` files — the scorer
classifies `answer_postprocess.py` + `s3_calibration.json` both missing
as `failure=E8 s3_artefacts_never_written` (S3=0, S4=0).

**No placeholders allowed.** Each `answer.json` must have `raw_model_output`
equal to the actual text produced by your local VLM for that image+question.
Values that are empty, equal to `unknown` / `n/a` / `none`, or start with
`heuristic:` / `fallback:` / `placeholder:` / `mock:` are treated as
placeholders. Any placeholder in the smoke batch will zero out that sample's
`parse_rate` contribution, and if run-level `placeholder_rate > 0.05` the
scorer caps S4 at 0.2 (and marks `inference_mode` as `heuristic` /
`placeholder` / `empty`).

**Required artefact 1 — `${WORKSPACE_DIR}/answer_postprocess.py`.** A Python
module that exposes a callable

```python
def postprocess(raw: str) -> str:
    """Return the short normalized answer string from raw VLM text."""
```

It must be importable without side effects (no downloads, no heavy imports
outside the function body), must never raise on any string input, and must
return a `str`. The scorer imports it and probes it with
`postprocess("no, the upper dermis shows only mild edema.")`. S4 **must**
`from answer_postprocess import postprocess` and write
`predicted_answer = postprocess(raw_model_output)` for every sample.

**How to write `answer_postprocess.py`.** Open-ended tasks (PathVQA /
VQA-RAD / SLAKE) — strip whitespace and trailing punctuation, lowercase,
then:
  1. **Yes/no collapse**: if the raw starts with or contains `yes` / `no`
     (optionally preceded by `answer:` / `the answer is`), return exactly
     `"yes"` or `"no"`.
  2. **Short-phrase extraction**: otherwise take the first sentence or
     clause (split on `. `, `, `, `;`). Drop leading fillers like
     `the image shows`, `this is`, `it is`, `answer:`.
  3. Return the shortest meaningful noun phrase (1–3 words is typical for
     these datasets).

MCQ tasks (MedXpertQA-MM / MedFrameQA) — parse the first standalone
letter in `A-E` (regex `\b([A-E])\b`) from the raw text; return the
uppercase letter. Defensive fallback: scan for `option X` / `answer: X` /
`(X)` patterns.

Never return empty string — fall back to the raw text trimmed to ≤ 64
chars if nothing matched, so S4 still has something to write.

**Required artefact 2 — `${WORKSPACE_DIR}/s3_calibration.json`.** A JSON list
with **>= 10 records** (15 preferred), one per calibration sample. Required
keys: `question_id`, `raw_model_output`, `predicted_answer`. Optional keys:
`gold_answer`, `hit`.

```json
{
  "question_id": "<id>",
  "raw_model_output": "<full decoded text>",
  "predicted_answer": "<postprocess(raw_model_output)>",
  "gold_answer": "<reference answer — OPTIONAL>",
  "hit": true
}
```

**About `gold_answer` / `hit` (OPTIONAL).** The public manifest intentionally
does **not** contain gold answers (privacy contract). You only have gold
if the dataset exposes it separately (e.g. HuggingFace dataset loader,
`data/<task>/public/<qid>/hint_answer.json`, or an external leaderboard).

- If you have real gold for some samples: include `gold_answer` and
  set `hit` to the boolean result of your postprocess vs. gold (lowercase
  + strip punctuation is the scorer's rule). Use the observed hit rate to
  iterate on `answer_postprocess.py` before S4 — if hit rate < 0.20, fix
  the post-processor first.
- If you have no gold: **omit both fields**. Do NOT fabricate
  `gold_answer="unknown"` / `""` / `"n/a"` to pad hits. The scorer skips
  gold checks on records without `gold_answer`.

**Anti-cheat rules (enforced by scorer):**

- When `gold_answer` IS present, it must be a real reference (not `""`,
  `"unknown"`, `"n/a"`, `"na"`, `"none"`, `"null"`, `"?"`, `"-"`). The
  scorer caps S3 at 0.5 if `invalid_gold_rate > 0.2` among records that
  include `gold_answer`. Records without `gold_answer` are ignored by this
  check.
- `raw_model_output` must be the **full decoded VLM text**, length ≥ 5
  characters, and must not start with punctuation (`, . ; : ! ? -`). A
  fragment like `"-glass nuclei."` / `"oma."` / `"ically removed"` means your
  decode is dropping the real prefix — typical fix: decode only the
  newly-generated token range (`tokenizer.decode(out[0][input_len:],
  skip_special_tokens=True)`) or use `KeywordsStoppingCriteria(conv.sep)` for
  LLaVA-Med's Mistral chat template. Fix decode, rerun S2 smoke, then
  rebuild `s3_calibration.json`. Scorer rejects the run with `S3 cap 0.5`
  when `malformed_raw_rate > 0.2`.

**Scorer enforcement:**
- `answer_postprocess.py` missing / not importable / missing the callable →
  `postprocess_valid=False` → S3 capped at 0.5 and `failure=E3`.
- `s3_calibration.json` missing / < 10 records / malformed records → same cap.
- `invalid_gold_rate > 0.2` (fake `"unknown"` gold, among records that
  include `gold_answer`) → same cap.
- `malformed_raw_rate > 0.2` (truncated / punctuation-prefixed raw) → same cap.
- Placeholder patterns in the calibration batch → S3 cap 0.2 (fake-output).
