# v5 S1–S5 Scoring Rubric (canonical)

The formal definition of the agentic axis. Authored by the benchmark lead; this file is the single source of truth. `scripts/compute_agentic_score.py` implements the **harness-deterministic rows only** (S4 + S5); **S1–S3 require an LLM judge** that reads each cell's `workspace/plan/`, `workspace/process/messages.json`, and `workspace/process/tool_log.jsonl`. Until the judge is wired up, S1–S3 remain `None` stubs and the workflow-score formula redistributes the weight over the remaining active steps (same convention as `eval_seg/aggregate.py`).

```
Overall Score = 0.50 · Agentic + 0.50 · Clinical
Agentic Score = Σ w[k] · step_scores[k]  /  Σ w[k]         (over non-None k)
                where w = {s1: 0.25, s2: 0.15, s3: 0.35, s4: 0.15, s5: 0.10}
```

---

## S1 Plan (25%) — 6 binary sub-criteria, averaged

| sub  | criterion                                                                                                              | codomain | notes                         |
|------|------------------------------------------------------------------------------------------------------------------------|----------|-------------------------------|
| s1a  | Created `plan.md`?                                                                                                     | {0, 1}   |                               |
| s1b  | Plan has clear pipeline (preprocess / inference / postprocess) instructions?                                           | {0, 1}   | 0 if no `plan.md`             |
| s1c  | Chose a real enhancement method (NOT identity-copy, NOT pure bicubic / gaussian bypass)?                               | {0, 1}   | 0 if no `plan.md`             |
| s1d  | Researched ≥ 3 candidate methods?                                                                                      | {0, 1}   | Lite: always 1                |
| s1e  | Created plan visualization (`plan.png`)?                                                                               | {0, 1}   | Lite: always 1                |
| s1f  | Plot has a clear pipeline diagram?                                                                                     | {0, 1}   | 0 if no plot; Lite: always 1  |

```
S1 = (s1a + s1b + s1c + s1d + s1e + s1f) / 6
Source: LLM judge
```

---

## S2 Setup (15%) — 5 binary sub-criteria, averaged

| sub  | criterion                                                                                                              | codomain |
|------|------------------------------------------------------------------------------------------------------------------------|----------|
| s2a  | Model checkpoint / weights downloaded? (LDCT lite: DRUNet via deepinv. MRI-SR lite: `caidas/swin2SR-classical-sr-x2-64`) | {0, 1} |
| s2b  | Compatibility check done? Verified framework versions, model dtype matches input dtype, no training patterns (`optim`, `loss`, `.train()`, `.backward()`) triggered | {0, 1} |
| s2c  | Environment setup succeeded?                                                                                           | {0, 1}   |
| s2d  | Env failures ≤ 5 attempts?                                                                                             | {0, 1}   |
| s2e  | Model loaded on GPU (not CPU fallback)?                                                                                | {0, 1}   |

```
S2 = (s2a + s2b + s2c + s2d + s2e) / 5
Source: LLM judge
```

---

## S3 Validate (35%) — 4-level discrete

Pick the single highest level the cell reaches.

| level | criterion                                                                                                              |
|-------|------------------------------------------------------------------------------------------------------------------------|
| 0.0   | No validation detected — went straight to batch inference                                                              |
| 0.2   | Shape-only — verified `enhanced.npy` shape matches spec (LDCT: 512×512 float32; MRI-SR: 720×512 float32) and nothing else |
| 0.5   | Range-aware — verified shape AND output value range is plausible (LDCT: HU range ~[−1300, +3200] preserved; MRI-SR: roughly [0, 1]; no NaN / inf / all-zero / all-constant outputs) |
| 1.0   | Single-patient pilot — ran full pipeline on ≥ 1 patient before batch, AND inspected output for shape AND range AND a sanity check on enhancement quality (input-vs-output visualization, OR measured noise-std reduction for LDCT, OR measured resolution doubling for MRI-SR) |

```
S3 ∈ {0.0, 0.2, 0.5, 1.0}
Source: LLM judge
```

---

## S4 Inference (15%) — continuous, harness-deterministic

```
S4 = 0.50 * completion_rate + 0.50 * format_valid

completion_rate = patients_with_enhanced_npy / total_patients         ∈ [0.0, 1.0]
format_valid    = 1.0 if ALL enhanced.npy pass shape/dtype/finite check, else 0.0
Source: harness (deterministic)
```

---

## S5 Submit (10%) — discrete {0.0, 0.5, 1.0}, harness-deterministic

```
S5 = 0.50 * has_valid_results + 0.50 * format_valid

has_valid_results = 1 if any patient scored AND mean_psnr is not NaN, else 0
format_valid      = 1 if all enhanced.npy pass format check, else 0
Source: harness (deterministic)
```

---

## Implementation status in this branch

| step | who scores it | current status in v5                                                                                 |
|------|---------------|-------------------------------------------------------------------------------------------------------|
| S1   | LLM judge     | **Not yet wired up** → stored as `None`                                                              |
| S2   | LLM judge     | **Not yet wired up** → stored as `None`                                                              |
| S3   | LLM judge     | **Not yet wired up** → stored as `None`                                                              |
| S4   | harness       | ✅ computed by `scripts/compute_agentic_score.py` exactly per the formula above                      |
| S5   | harness       | ✅ computed by `scripts/compute_agentic_score.py` exactly per the formula above                      |

When the LLM judge is wired up, it will populate S1/S2/S3 for each cell, the redistribution in `compute_workflow_score()` will automatically stop redistributing, and every cell's `agentic_score` / `overall_score` will be re-emitted with no other code change.

All the raw inputs the LLM judge needs (`plan.md`, `plan.png` if produced, `messages.json`, `tool_log.jsonl`, the per-turn `trace.jsonl`) are already written by `agent_loop.py` into `workspace/plan/` and `workspace/process/` of every run.
