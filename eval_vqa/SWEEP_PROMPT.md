# VQA Benchmark Sweep — Portable Run Prompt

## Task

Run `eval_vqa` benchmark sweep covering **6 agents × 3 tasks × 10 samples** on
`tier=lite`. Historical runs with `rating ∈ {A,B}` and `smoke=True` are
reusable — do **not** rerun those.

## Agents

```
glm-5, kimik2.5, gpt-5.4, minimax-m2.5, gemini-3.1-pro, claude-opus-4-6
```

## Tasks

```
pathvqa-task, vqa-rad-task, slake-task
```

## Metrics to report

- `overall  = 0.5 × workflow + 0.5 × task_score`
- `agentic` = workflow score = Σ wᵢ sᵢ with S1 0.25, S2 0.15, S3 0.35, S4 0.15, S5 0.10
- `task`    = `evaluation.accuracy` (open-ended: judge; MCQ: exact)
- `time_s`  = `runtime.wall_time_s`
- `cost`    = `runtime.estimated_cost_usd`
- `tokens`  = `runtime.total_tokens`
- `fail_rate` = fraction of expected samples that produced no valid
  answer, averaged across runs (per-run value is `1 - completion_rate`
  from `eval_report.aggregate.fail_rate`; matches eval_seg's
  `1 - inference_completes`). Target: 5–30%.

All of these come from each run's `outputs/run_summary.json`.

## Cells to actually run (13 runs)

- **vqa-rad** × 6 agents  (Sweep A)
- **slake**   × 6 agents  (Sweep B, including gemini rerun — prior was smoke=False)
- **pathvqa** × `claude-opus-4-6` only (Sweep C; other 5 agents reuse history)

## History to reuse (do not rerun)

| Agent | Task | Source run dir |
|---|---|---|
| glm-5 | pathvqa | `runs/vqa2-rerun4-judge-sonnet-183334/glm-5-r01/` |
| kimik2.5 | pathvqa | `runs/vqa2-rerun4-judge-sonnet-183334/kimik2.5-r01/` |
| gpt-5.4 | pathvqa | `runs/vqa2-pathvqa-lite-gpt54-p4-260419-052242/` |
| minimax-m2.5 | pathvqa | `runs/vqa2-bug044045-verify-260420-040223/minimax-m2.5-r01/` |
| gemini-3.1-pro | pathvqa | `runs/vqa2-smoke-gemini-260419-082533/gemini-3.1-pro-r01/` |

## Preconditions

- `NVDA_API_KEY` set (LLM judge; most agents also route through NVDA Inference API)
- Optional: `OPENROUTER_API_KEY` for `kimik2.5`
- `/dev/shm` ≥ 8G (≥ 12G if `--parallel-workers 3`)
- LLaVA-Med 7B weights downloadable (first run fills `--shared-hf-cache`)
- `cd <repo>/eval_vqa`

## Commands

```bash
TS=$(date +%y%m%d-%H%M%S)
ROOT=<repo>/runs
export VQA_ANSWER_JUDGE=1
export ANSWER_JUDGE_BASE_URL=https://inference-api.nvidia.com
export ANSWER_JUDGE_MODEL=aws/anthropic/bedrock-claude-sonnet-4-6

# A — vqa-rad (6 agents)
nohup python sweep.py --task vqa-rad-task --tier lite --subset all \
  --agents glm-5,kimik2.5,gpt-5.4,minimax-m2.5,gemini-3.1-pro,claude-opus-4-6 \
  --repeats 1 --parallel-workers 3 --gpu-devices 0,1,2 --sample-limit 10 \
  --sweep-root $ROOT/vqa2-sweepA-vqarad-$TS \
  --shared-hf-cache <repo>/hf_cache_shared \
  > $ROOT/vqa2-sweepA-vqarad-$TS.log 2>&1 &

# B — slake (6 agents, incl. gemini rerun)
nohup python sweep.py --task slake-task --tier lite --subset all \
  --agents glm-5,kimik2.5,gpt-5.4,minimax-m2.5,gemini-3.1-pro,claude-opus-4-6 \
  --repeats 1 --parallel-workers 3 --gpu-devices 0,1,2 --sample-limit 10 \
  --sweep-root $ROOT/vqa2-sweepB-slake-$TS \
  --shared-hf-cache <repo>/hf_cache_shared \
  > $ROOT/vqa2-sweepB-slake-$TS.log 2>&1 &

# C — pathvqa claude-opus-4-6 only
nohup python sweep.py --task pathvqa-task --tier lite --subset all \
  --agents claude-opus-4-6 \
  --repeats 1 --parallel-workers 1 --gpu-devices 0 --sample-limit 10 \
  --sweep-root $ROOT/vqa2-sweepC-opusPath-$TS \
  --shared-hf-cache <repo>/hf_cache_shared \
  > $ROOT/vqa2-sweepC-opusPath-$TS.log 2>&1 &
```

Run serially if GPU count is small. Monitor with `tail -F`.

## Aggregation script (after sweeps finish)

```python
import json, glob, os
ROOT = "<repo>/runs"
STEP_W = {"s1":0.25,"s2":0.15,"s3":0.35,"s4":0.15,"s5":0.10}
rows = []
for rs in glob.glob(f"{ROOT}/vqa2-sweep*/*/outputs/run_summary.json"):
    d = json.load(open(rs))
    ev = d.get("evaluation") or {}
    rt = d.get("runtime") or {}
    ss = ev.get("step_scores") or {}
    agentic = sum(STEP_W[k] * (ss.get(k) or 0) for k in STEP_W)
    task    = ev.get("accuracy") or 0.0
    overall = 0.5 * agentic + 0.5 * task
    rows.append({
        "agent":   d.get("agent"),
        "task":    d.get("task"),
        "overall": round(overall, 4),
        "agentic": round(agentic, 4),
        "task":    round(task,    4),
        "time_s":  rt.get("wall_time_s"),
        "cost":    rt.get("estimated_cost_usd"),
        "tokens":  rt.get("total_tokens"),
        "rating":  ev.get("rating"),
    })

# fail_rate is per-run (1 - completion_rate); average across runs
fail_rate = sum(float(ev.get("fail_rate", 0.0)) for ev in evs) / max(len(evs), 1)
print(f"fail_rate = {fail_rate:.2%}  (n={len(rows)})")
# print markdown table ...
```

## Known pitfalls

1. **/dev/shm too small** → `Bus error` during LLaVA-Med load → drop to
   `--parallel-workers 2` or run under Docker with `--shm-size=8g`.
2. **gpt-5.4 S2 smoke** often writes `raw_output_sample="."` (1 char) →
   `inference_verifier` requires ≥ 5 chars → run tagged E2.
3. Agent names: `kimik2.5` / `glm-5` (no hyphen between letters and digits for
   kimi; `-` between `glm` and `5`).
4. `answer_judge` 401 → ensure the env key matches the judge `BASE_URL`
   (NVDA key must go with `inference-api.nvidia.com`, not OpenRouter).
5. LLaVA-Med empty decode → S2 must use `conv_templates`, `DEFAULT_IMAGE_TOKEN`
   /`tokenizer_image_token`, and slice decode to `output_ids[:, input_ids.shape[-1]:]`.

## Shape of final output

```markdown
| Agent | Task | Overall | Agentic | Task | Time (s) | Cost (USD) | Tokens | Rating |
|---|---|---|---|---|---|---|---|---|
| glm-5 | pathvqa | 0.55 | 0.93 | 0.40 | 1118 | 1.29 | 2.06M | A |
| ... |
```

Plus a single `fail_rate` number across the whole 18-cell matrix (13 fresh + 5
reused).
