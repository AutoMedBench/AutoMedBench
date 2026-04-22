# MedAgentsBench VQA Benchmark (V2)

Medical visual question answering (VQA) agent benchmark. Mirrors `eval_seg/`
and follows the **single-LLM / execute_code-only** coding-agent paradigm: the
agent writes its own Python across one long conversation to complete all of
S1–S5; the harness provides no auxiliary tools.

Datasets covered: **PathVQA**, **VQA-RAD**, **SLAKE-EN**, **MedFrameQA**,
**MedXpertQA-MM**.

> 中文版本见 [`README_zh.md`](README_zh.md).

---

## 1. Quick start

### 1.1 Environment

```bash
conda activate base
cd eval_vqa
pip install -r requirements.txt
```

Required environment variables: `NVDA_API_KEY` or `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` (agent-dependent), `HF_TOKEN` (for gated model downloads),
`OPENROUTER_API_KEY` (LLM-as-judge, defaults to Claude Haiku 4.5).

### 1.2 Single benchmark run

```bash
python -u eval_vqa/benchmark_runner.py \
  --agent nvda-claude-opus-4-6 \
  --task pathvqa-task \
  --tier lite \
  --subset all \
  --output-dir ./runs/<experimenter>/vqa-<short>-<task>-<tier>
```

- `--task` ∈ `pathvqa-task` / `vqa-rad-task` / `slake-task` /
  `medframeqa-task` / `medxpertqa-mm-task`
- `--tier` ∈ `lite` (fixed model `microsoft/llava-med-v1.5-mistral-7b`) /
  `standard` (five candidate open VLMs)
- `--subset` ∈ `all` / `smoke` (10 samples) / `calibration` (15 samples)
- `--output-dir` auto-appends `<YYMMDD>-<6hex>` so parallel runs never collide

### 1.3 Scoring only (existing predictions)

```bash
python eval_vqa/run_eval.py \
  --gt-dir    <staged_private>/<qid>/ \
  --agent-dir <run_root>/predictions/ \
  --public-dir <staged_public>/ \
  --task pathvqa-task --tier lite \
  --enable-answer-judge \
  --answer-judge-model anthropic/claude-haiku-4.5
```

### 1.4 Sweep (multi-agent × repeat × GPU)

```bash
python eval_vqa/sweep.py \
  --agents nvda-claude-opus-4-6,gpt-5.4,gemini-3.1-pro \
  --repeats 3 --parallel-workers 4 \
  --gpu-devices 0,1,2,3 \
  --task pathvqa-task --tier lite \
  --sweep-root /data2/<experimenter>/vqa-sweep-<name>
```

**`/dev/shm` preflight** (BUG-044-derived): when `--parallel-workers > 1`,
sweep checks `/dev/shm` free space against `max(4 GiB, 4 × workers)`. If
insufficient, it aborts with three mitigations: reduce workers, run under
docker with `--shm-size=8g`, or pass `--allow-low-shm` to override.

---

## 2. Layout

```
eval_vqa/
├── benchmark_runner.py      # Agent-loop driver (single LLM + execute_code)
├── run_eval.py              # Scoring-only entry
├── sweep.py                 # Parallel sweep driver (with shm preflight)
├── aggregate.py             # workflow_score / step_scores / rating
├── vqa_scorer.py            # Sample-level EM / F1 / yes-no / judge scoring
├── answer_judge.py          # LLM-as-judge (Claude Haiku via OpenRouter)
├── llm_judge.py             # Workflow-level S1-S3 heuristic judge
├── failure_classifier.py    # E1-E8 error-code classifier
├── inference_verifier.py    # smoke_forward / model_call / postprocess checks
├── tool_call_recovery.py    # minimax/qwen XML tool_call recovery (BUG-043/040)
├── format_checker.py        # Submission schema validator
├── tier_config.py           # lite / standard differences + step_weights
├── task_loader.py           # tasks/*/config.yaml + model_info.yaml loader
├── stage_hf_datasets.py     # Public/private manifest split (privacy contract)
├── medal_tier.py            # Medal tiering
├── detail_report.py         # Human-readable report
├── prompts/                 # Base prompts per stage (see §3)
│   ├── s1_plan/{lite,standard}.md
│   ├── s2_setup/lite.md
│   ├── s3_validate/lite_standard.md
│   ├── s4_inference/all.md
│   └── s5_submit/all.md
├── tasks/<task-id>/         # Task config + per-stage skill snippets
│   ├── config.yaml          # answer_mode / valid_labels / subsets …
│   ├── model_info.yaml      # lite_model / standard_candidates
│   ├── lite_s1.md / lite_s2.md / lite_s3.md
│   └── standard_s1.md / standard_s3.md
└── tests/                   # pytest suite (100+ cases)
```

---

## 3. Five-stage workflow

The agent progresses S1 → S5 within one long conversation, using only
`execute_code`. Stage weights: **S1 0.25 / S2 0.15 / S3 0.35 / S4 0.15 /
S5 0.10**.

Each stage below lists: **definition → artefact contract → full prompt →
scoring rules**.

### 3.1 S1 — PLAN (weight 0.25)

**Definition**: produce `plan/plan.md` covering model selection, data
contract, label/answer extraction, and the smoke plan.

**Prompt (lite)**:

```markdown
Use `UCSC-VLAA/MedVLThinker-3B-RL_m23k`. Write `plan.md` covering:

- local model loading path
- staged `question.json` / `answer.json` contract
- label extraction into `predicted_label`
- smoke validation on `1-10` samples before any larger run
```

**Prompt (standard)**: the agent picks from five candidates (MedVLThinker /
MedVLSynther / Qwen2.5-VL-3B / gemma-4-E2B / gemma-4-E4B), must document
rejected candidates, and must flag gated-access risk.

**Scoring**: `None` by default; populated by `llm_judge.py` when
`--llm-judge` is set (always on in `benchmark_runner.py`). When no judge
runs, S1 is excluded from the workflow denominator (renormalization;
review fix #2).

### 3.2 S2 — SETUP (weight 0.15)

**Definition**: install dependencies, download model, run one GPU forward
pass, write `smoke_forward.json`.

**Prompt**:

```markdown
Install dependencies from the task `requirements.txt`. For PyTorch packages,
use the official wheel index and choose a CUDA build that matches the host
driver instead of assuming the newest wheel will run. Then download model and
processor assets locally, and confirm at least one successful forward pass on
GPU before continuing.

**Required artefact — `${WORKSPACE_DIR}/smoke_forward.json`.** After your
smoke forward pass, write this file with the following schema:

{
  "model_name": "<HF repo id you actually loaded>",
  "device": "cuda" | "cuda:0" | "...",
  "wall_s": <float, end-to-end seconds, must be >= 0.3>,
  "raw_output_sample": "<first ~200 chars of the raw decoded model text>",
  "success": true
}

The scorer reads this artefact directly. If `smoke_forward.json` is missing,
`success` is not `true`, `wall_s < 0.3`, or `raw_output_sample` is empty /
starts with `heuristic:` / `fallback:` / `placeholder:`, **S2 will be forced
to 0.0 regardless of any other evidence**.
```

**Scoring** (binary triplet, P3):

```
S2 = mean(env_ready, model_loaded, smoke_forward_passed)
```

- `env_ready`: at least one valid output or `completion_rate > 0`
- `model_loaded`: trace contains `model.generate()` / `from_pretrained()`
- `smoke_forward_passed`: `inference_verifier.check_smoke_forward()` passes

### 3.3 S3 — VALIDATE (weight 0.35, highest)

**Definition**: calibrate on 15 samples; produce `answer_postprocess.py`
(short-answer normalizer) + `s3_calibration.json` (gold comparison).

**Full prompt** (`prompts/s3_validate/lite_standard.md`):

```markdown
Run inference on **exactly 15 samples** from the public split as a calibration
pass. Design a deterministic post-processor before S4 starts at scale.

**Turn budget — commit, don't loop.** S3 calibration should take no more than
10 `execute_code` turns. Partial calibration is better than a stalled run with
zero `answer.json` files.

**Required artefact 1 — `${WORKSPACE_DIR}/answer_postprocess.py`**:

def postprocess(raw: str) -> str:
    """Return the short normalized answer string from raw VLM text."""

Must be importable without side effects, never raise, return str.

**Required artefact 2 — `${WORKSPACE_DIR}/s3_calibration.json`** (≥15 records):

{"question_id": "<id>",
 "raw_model_output": "<full decoded text>",
 "predicted_answer": "<postprocess(raw_model_output)>",
 "gold_answer": "<reference answer>",
 "hit": true|false}

**Anti-cheat rules (enforced):**
- `gold_answer` must come from a real reference; writing "unknown" / "n/a" /
  "" to fabricate a hit triggers `invalid_gold_rate > 0.2 → S3 cap 0.5`.
- `raw_model_output` length ≥ 5, must not start with punctuation;
  `malformed_raw_rate > 0.2 → S3 cap 0.5` (BUG-042, typical LLaVA-Med decode
  truncation symptom).

**Scorer enforcement:**
- `answer_postprocess.py` missing / not importable / no callable →
  `postprocess_valid=False` → S3 cap 0.5, `failure=E3`.
- `s3_calibration.json` missing / <15 records / malformed → same cap.
- 15-turn warning (BUG-044): if either artefact is still missing, the agent
  receives `[S3 BUDGET WARNING]` nudging it to commit and proceed to S4.
```

**Scoring**:
- `None` by default (needs `llm_judge.py` to populate)
- `postprocess_valid=False` → hard cap 0.5
- `inference_mode ∈ {heuristic, placeholder, empty}` → cap 0.2 (prevents
  the judge from inflating scores on fake outputs)

### 3.4 S4 — INFERENCE (weight 0.15)

**Definition**: run real VLM inference across all question_ids; write one
`<qid>/answer.json` per question; load the model exactly once.

**Prompt highlights** (full text in `prompts/s4_inference/all.md`):

- **Load the model once**: no per-sample `from_pretrained`, no subprocess
  reload loops.
- **Real inference required**: `raw_model_output` must come from
  `.generate()`; `heuristic:` / `fallback:` / `placeholder:` / `unknown` /
  empty strings are treated as placeholders.
- **Use the S3 post-processor**:
  `from answer_postprocess import postprocess` is mandatory; do not
  reinvent normalization in S4.
- **Open-ended short-answer contract** (BUG-047 + review #7): for tasks
  with `answer_mode=open_ended` (PathVQA / VQA-RAD / SLAKE), the system
  prompt injects an "Open-ended answer contract" block:
  - `predicted_answer` ≤ 5 words; VLM prose must be reduced by postprocess.
  - Yes/no: emit exactly `yes` or `no`.
  - No leading articles, trailing punctuation, or explanations.

**Scoring** (P1-B real-but-broken aware):

```
base = 0.5 · completion_rate + 0.5 · parse_rate

Caps (applied in order):
- placeholder_rate > 0.05     → cap 0.2
- model_call_detected = False → cap 0.3
- completion ≥ 0.99 ∧ placeholder ≤ 0.05 ∧ model_call
  ∧ accuracy < 0.05           → cap 0.5  (real_but_broken)
```

### 3.5 S5 — SUBMIT (weight 0.10)

**Definition**: final verification of completeness and schema validity.

**Prompt**:

```markdown
Before submission, verify completeness, parseability, and schema validity for
all expected prediction records.
```

**Scoring**:

```
S5 = 0.5 · has_valid_results + 0.5 · submission_format_valid
```

- `has_valid_results`: at least one valid output
- `submission_format_valid`: `format_checker.check_submission()` passes

---

## 4. Overall score and rating

### 4.1 Formulas

```
workflow_score = Σ w_i · step_i   (over active / non-None steps; review #2)
                 ────────────────
                 Σ w_i
                   i ∈ active_steps

task_score = accuracy_judge         (open-ended + --enable-answer-judge)
           = 0.5·EM + 0.5·F1        (open-ended without judge; yes/no strict)
           = exact-label accuracy   (multiple_choice)

overall = 0.5 · workflow_score + 0.5 · task_score
```

### 4.2 Medals and ratings

`medal_tier.py` assigns `tier ∈ {0, 1, 2}` from `task_score`.

```
rating = F   if not submission_format_valid
         or valid_outputs == 0
         or completion_rate < 0.5
       = A   if tier >= 2     (gold)
       = B   if tier >= 1     (silver/bronze, resolved)
       = C   otherwise        (ok but not resolved)
```

### 4.3 Weight table

| Stage | Weight | Scoring source |
|---|---|---|
| S1 Plan | 0.25 | `llm_judge.py` (optional) |
| S2 Setup | 0.15 | Ternary binary (env / model / smoke) |
| S3 Validate | **0.35** | `llm_judge.py` + postprocess caps |
| S4 Inference | 0.15 | completion + parse + guards |
| S5 Submit | 0.10 | valid + format |

---

## 5. Answer modes

`answer_mode` in a task's `config.yaml` selects the scoring branch.

| answer_mode | Tasks | task_score |
|---|---|---|
| `multiple_choice` | medxpertqa-mm / medframeqa | exact label (A–E) |
| `open_ended` | pathvqa / vqa-rad / slake | LLM judge (default) or 0.5·EM+0.5·F1 |

Yes/no subsets (gold ∈ {yes, no}) use strict `yes_no_accuracy` so
`"yes, the cyst wall ..."` does not get inflated via token F1.

---

## 6. LLM-as-judge (BUG-038)

**When active**: `answer_mode=open_ended` and either `--enable-answer-judge`
or `VQA_ANSWER_JUDGE=1`. `benchmark_runner.py` enables it by default.

**Model**: `anthropic/claude-haiku-4.5` via OpenRouter; override with
`ANSWER_JUDGE_MODEL` / `ANSWER_JUDGE_BASE_URL`.

**Score buckets**: 0.0 / 0.5 / 1.0 (clinical incorrect / partial / equivalent).

**Cache**: `<workspace>/answer_judge_cache.jsonl`, keyed on
`sha256(qid∥gold∥pred∥model)`. **Concurrent sweep writes are serialized via
`fcntl.LOCK_EX`** (review #1) to prevent interleaved JSONL lines.

**Fallback**: missing API key or backend error → token-F1 / yes-no heuristic,
`judge_backend` field tagged `heuristic_fallback`; `run_eval.py` emits a
stderr WARNING (review #5) so operators know the judge column is partially
heuristic.

---

## 7. Privacy contract

**Enforced**: during S1–S4 the agent only reads `question.json` under the
public directory. `answer*` / `reasoning_chain` / `correct_answer` /
`reference_answer` fields are forbidden. Gold answers are loaded only at
S5 scoring time from `<gt_dir>/<qid>/answer.json`.

`stage_hf_datasets.py` enforces the split: `pub_root/<qid>/question.json`
contains only `question_id / dataset / question / image_paths / split /
medical_task / body_system / question_type`; `prv_root/<qid>/answer.json`
holds `answer_label / answer_text`. Covered by
`test_placeholder_detection.py` and related tests.

---

## 8. Failure codes

Produced by `failure_classifier.py` as `primary_failure`:

| Code | Meaning | Trigger |
|---|---|---|
| E1 | Hallucination | Answer mismatched image / question (semantic) |
| E2 | Resource error | smoke_forward failed / env not ready |
| E3 | Logic error | postprocess not importable / malformed calibration |
| E4 | Code error | execute_code raised and never recovered |
| E5 | Format error | `model_call=False` or placeholder/empty outputs |
| **E8** | S3 artefacts never written | smoke passed but both `answer_postprocess.py` and `s3_calibration.json` are missing and no `answer.json` exists (BUG-044) |

---

## 9. Sandbox / isolation

Agent filesystem view:

- Read-only: `/data/public/`
- Read-write: `/workspace/run_<id>/`
- Blocked: `/data/private/`, `eval_vqa/runs/`, sibling run dirs, harness code

Docker isolation (recommended for parallel sweeps):

```bash
bash eval_vqa/run_vqa_docker.sh
# or
python eval_vqa/docker/orchestrator.py --agent ... --gpu-id N
```

`/dev/shm` should be ≥ 8 GiB for parallel LLaVA-Med 7B loads. Typical Bus
error symptom: `inference_mode=empty` with S2/S3 dying mid-run (noted in
CLAUDE.md).

---

## 10. Development and testing

```bash
cd eval_vqa
python -m pytest tests/ -v
```

**103 tests passing** after the review round. Coverage:

- `test_scoring.py` / `test_workflow_score_renorm.py` — aggregation and renorm
- `test_answer_judge.py` — LLM judge + concurrent cache
- `test_tool_call_recovery.py` — BUG-043/040 XML recovery + prose rejection
- `test_failure_classifier_e8.py` — E8 S3 stall
- `test_placeholder_detection.py` — privacy / placeholder heuristics
- `test_sweep_preflight.py` — `/dev/shm` preflight
- `test_prompt_open_ended_guidance.py` — open-ended short-answer contract
- `test_workspace_substitution.py` — workspace placeholder (BUG-046)
- `test_length_finish_rate.py` — finish_reason=length diagnostics (BUG-045)

---

## 11. Recent fixes (2026-04 review round)

| # | Issue | Fix |
|---|---|---|
| 1 | `answer_judge` cache concurrency race | `fcntl.LOCK_EX` + `fsync`, one atomic write per entry |
| 2 | S1/S3 None steps silently zeroed workflow | `compute_workflow_score` renormalizes over active steps |
| 3 | `tool_call_recovery` prose false-positives | Require wrapper sentinel or line-start anchor; prose `<invoke>` rejected |
| 4 | No `/dev/shm` preflight | `sweep.preflight_shm()` + `--allow-low-shm` override |
| 5 | Judge heuristic fallback was silent | `judge_fallback_count` field + `run_eval` stderr WARNING |
| 6 | Turn-15 S3 warning only checked `pp.py` | Also checks `s3_calibration.json`; names missing artefact(s) |
| 7 | No open-ended short-answer contract | `build_tier_system_prompt` injects a ≤5-word contract when `answer_mode=open_ended` |

Historical bug tickets: `bug_issues/BUG-038` … `BUG-047`. Scan the relevant
ticket before editing the corresponding area.

---

## 12. Notes

- Do **not** mix the seg `agent_config.yaml` with the VQA one; the VQA suite
  uses provider-qualified keys like `nvda-<agent>`.
- **Tier only affects prompts**, not scoring. `lite` and `standard` share
  `_WEIGHTS`.
- `old_vqa/` and `vqa_hard/` are historical snapshots — read-only, do not
  extend.
- All run artefacts go under `runs/<experimenter>/...`; never read another
  experimenter's directory as ground truth, never stage it into git.
