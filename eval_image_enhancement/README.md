# eval_image_enhancement (v5)

Medical image enhancement benchmark for coding agents — sibling domain to `eval_seg/`, `eval_vqa/`. An agent must autonomously plan, set up, validate, and run inference on 100 2D medical images per cell, producing enhanced outputs that are scored against held-out reference images in an isolated evaluation container.

**v5 = full 2-container architecture (agent + eval, separate images, orchestrator chains them), matching the `eval_seg` gold standard.** Previous iterations (v1–v4) are not included in this branch.

---

## 1. Quick facts

| | |
|---|---|
| Tasks | `ldct-denoising-task` (LDCT low-dose CT denoising), `mri-sr-task` (MRI ×2 super-resolution) |
| Tiers | `lite` (required DNN method specified) and `standard` (agent chooses from candidate list) |
| Dataset | 100 patients per task (LDCT_SimNICT, MRI_SR_SRMRI — 2D slices) |
| Agents | 6 production models. `orchestrator.py` reads `eval_seg/agent_config.yaml` (shared across domains — this branch does not ship a duplicate) |
| Repeats | 10 per cell in the full matrix → 240 runs total |
| Per-case metric | `pass_rate` — fraction of patients with `SSIM ≥ classical_baseline − 0.02` |
| Aggregate metric | `rating` ∈ {A, B, C, F} against dataset-calibrated PSNR + SSIM bands |
| Inference-only | `.backward()`, `optimizer.step()`, `model.train()`, `torch.optim.*` all blocked by sandbox |

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Host orchestrator (docker/orchestrator.py)                       │
│   • Render tier prompt with container-internal paths             │
│   • Stage public data per run                                    │
└───┬──────────────────────────────────────────────┬──────────────┘
    │                                              │
    ▼                                              ▼
┌────────────────┐                         ┌────────────────┐
│ Agent container│                         │ Eval container │
│ ie-agent:v5    │                         │ ie-eval:v5     │
│ GPU + network  │      ──finishes──►      │ GPU + no-net   │
│ /data/public   │                         │ /data/private  │
│ /workspace     │                         │ /agent_outputs │
│ sandbox layer3 │                         │ /results       │
└────────────────┘                         └────────────────┘
```

**Isolation guarantees (both containers):**
- `--read-only` rootfs + `--tmpfs /tmp`
- `--security-opt no-new-privileges`
- matched host UID/GID — no root-owned output files
- `--pids-limit 4096`

**Agent container:**
- GPU access (`--gpus device=<id>`)
- network `bridge` (needs LLM API)
- mounts public data read-only, workspace read-write
- `/data/private`, `/eval`, `/results`, `/bands` **not mounted** (verified at entrypoint)
- three-layer sandbox (Python path here is tight; bash path is narrower):
  - Static substring / regex block on every submitted snippet before execution
  - **Python** scripts get a `sys.addaudithook()` preamble injected at runtime that hard-kills (`_os._exit(99)`) any `open` / `os.listdir` / `os.scandir` / `os.chdir` / `os.mkdir` / `os.rename` / `os.remove` / `os.symlink` / `shutil.copy*` / `shutil.rmtree` call touching `/data/private`, `/eval/`, `/results/`, `/bands/`, `*/reference.npy`, `*/ground_truth.csv`, `*/baseline_bands.json`, or the agent's own harness
  - **Bash** scripts only get function wrappers on `cat / head / tail / less / more / cp / mv / ln`; other readers (awk, sed, python -c, arbitrary binaries) are not intercepted at shell level. Defense-in-depth still applies because the forbidden paths are not mounted into the container at all — the shell wrappers are a secondary tripwire, not the main isolation mechanism
- 1 violation → warning; 2nd violation → `exit 99` → orchestrator zeros all step scores, rating F

**Eval container:**
- `--network none` (defense-in-depth against exfil)
- GPU for LPIPS (~50× faster than CPU; network is still blocked)
- no agent code
- writes `/results/detail_report.json`

## 3. Scoring (v5)

Mirrors `eval_seg/aggregate.py` exactly (same step weights, same overall composition) so agentic numbers are comparable across sibling domains.

```
                           Overall Score
                       (50% Agentic + 50% Clinical)
                      /                            \
           Agentic Score                        Clinical Score
      (weighted avg of S1-S5)             (per-task normalized
                                           PSNR/SSIM/LPIPS, see §3.b)
```

**3.a — Agentic score (S1–S5):**

The canonical per-step rubric is specified in [`RUBRIC.md`](RUBRIC.md) — S1 is a 6-item checklist, S2 is a 5-item checklist, S3 is a 4-level discrete ladder, S4 and S5 are continuous formulas. Weights are `{s1: 0.25, s2: 0.15, s3: 0.35, s4: 0.15, s5: 0.10}`, matching `eval_seg/aggregate.py`.

| step | weight | source   | status in this branch |
|------|--------|----------|-----------------------|
| S1 PLAN      | 0.25 | LLM judge | rubric written (see `RUBRIC.md` §S1); judge not yet wired → stored as `None` |
| S2 SETUP     | 0.15 | LLM judge | rubric written (see `RUBRIC.md` §S2); judge not yet wired → stored as `None` |
| S3 VALIDATE  | 0.35 | LLM judge | rubric written (see `RUBRIC.md` §S3); judge not yet wired → stored as `None` |
| S4 INFERENCE | 0.15 | harness   | `0.5 * completion_rate + 0.5 * format_valid` — computed |
| S5 SUBMIT    | 0.10 | harness   | `0.5 * has_valid_results + 0.5 * format_valid` — computed |

Null (None) steps are skipped and their weight redistributes over the remaining steps — identical to seg's `compute_workflow_score`. So every agentic number currently in `summary.csv` is `0.60 · S4 + 0.40 · S5` (the renormalized-over-S4+S5 formula). When the LLM judge lands, S1–S3 populate automatically with no code change. Isolation breach zeroes S4 and S5, giving `agentic_score = 0`.

**3.b — Clinical score (per-task, inference-quality axis):**

Per patient: **PSNR** (dB), **SSIM** (skimage gaussian), **LPIPS** (AlexNet). Shuffled negative control runs the same metrics with references shifted by one patient to confirm the scorer is not returning high numbers by default.

Then:
- **`pass_rate`** = fraction of patients with `SSIM ≥ classical_baseline_SSIM − 0.02` (primary metric)
- **`clinical_score`** = `(psnr_norm + ssim_norm + lpips_norm) / 3` with per-task linear-window normalization (see `enhancement_scorer.TASK_NORM`)
- **`rating`** in {A, B, C, F} against `baseline_bands.json` (dataset-calibrated):
  - A: beats top baseline (DRUNet / bicubic) by ≥ 1 dB PSNR **and** ≥ 0.02 SSIM
  - B: within ±1 dB / ±0.02 SSIM of top baseline
  - C: within ≤ 3 dB / ≤ 0.05 SSIM of top baseline (explicitly widened from v3 to avoid B/C collapse on LDCT)
  - F: below C, < 50% valid format, or isolation breach

Failures are explicitly classified. Definitions are precise:

| label           | rule (first match wins)                                                          |
|-----------------|----------------------------------------------------------------------------------|
| `pass_{A,B,C}`  | scorer assigned a letter rating (outputs on disk, bands met)                     |
| `F_breach`      | orchestrator saw `agent_exit_code == 99` (2nd sandbox violation → kill)          |
| `F_no_output`   | `n_valid == 0` — scorer found zero valid `enhanced.npy` files                    |
| `F_timeout`     | `n_valid > 0` **and** `submitted == false` — agent wrote at least one valid `enhanced.npy` but never called `submit_results`. Catch-all bucket: wall-clock cap, turn cap, API error, exec hang, repeated pre-submit rejection, etc. `n_valid` may be anywhere from 1 up to the full patient count |
| `F_below_band`  | submitted, format-valid, but below C band                                        |

Note: because the eval container scores whatever is in `/agent_outputs`, a run that writes 100 valid `enhanced.npy` files but forgets to call `submit_results` can still receive `pass_{A,B,C}` — the scorer does not require an explicit submit. `submit_results` only governs the 50% pre-submit gate. This is an intentional choice so that unrelated bugs in the agent's submit path do not discard otherwise-valid science.

## 4. Layout

```
eval_image_enhancement/
├── docker/
│   ├── agent/             # ie-agent:v5 (GPU + network)
│   │   ├── Dockerfile.agent
│   │   ├── entrypoint_agent.sh
│   │   ├── agent_loop.py           # LLM turn loop inside container
│   │   └── agent_code_executor.py  # 3-layer sandbox
│   ├── eval/              # ie-eval:v5 (GPU + --network none)
│   │   ├── Dockerfile.eval
│   │   ├── entrypoint_eval.sh
│   │   └── run_eval.py             # black-box scorer
│   ├── tests/             # offline sandbox regression tests (no LLM, no net)
│   │   ├── test_sandbox.py
│   │   ├── seg_redteam_corpus.json
│   │   └── README.md
│   └── orchestrator.py    # host-side: build, chain, collect
├── ldct-denoising-task/   # config + tier skills + baseline_bands.json
├── mri-sr-task/           # config + tier skills + baseline_bands.json
├── prompts/               # preamble / env / S1–S5 markdown, per tier
├── scripts/
│   ├── launch_matrix_v5.py       # Popen scheduler, 6 agents × 2 tiers × 2 tasks × N repeats
│   ├── compute_baseline_bands.py # builds baseline_bands.json once per dataset
│   ├── rescue_eval.py            # re-score cells whose eval container was killed
│   ├── aggregate_v5.py           # bootstrap 95% CI + failure breakdown → summary.csv/json
│   ├── plot_v5_summary.py        # heatmap / scatter / efficiency
│   └── plot_v5_bars.py           # pass-rate bars / stage timing / token fairness
├── enhancement_scorer.py   # PSNR/SSIM/LPIPS + rating + bands
├── format_checker.py
├── results/
│   └── matrix_v5_full/    # summary.csv, summary.json, per-cell reports
└── figures/               # final figures for matrix_v5_full
```

## 5. How to reproduce

```bash
# 1. Build both container images (host UID/GID propagated)
python eval_image_enhancement/docker/orchestrator.py --build-only

# 2. Single cell (debug)
python eval_image_enhancement/docker/orchestrator.py \
    --agent claude-opus-4-6 --task ldct-denoising-task --tier lite \
    --n-patients 100 --gpu-id 0 --repeat-idx 0 \
    --max-seconds 2700 --output-dir runs/single

# 3. Full matrix (240 runs across 8 GPUs, 10 repeats per cell)
python eval_image_enhancement/scripts/launch_matrix_v5.py \
    runs/matrix_v5_full 10 8 100

# 4. Aggregate + figures
python eval_image_enhancement/scripts/aggregate_v5.py runs/matrix_v5_full
python eval_image_enhancement/scripts/plot_v5_summary.py runs/matrix_v5_full
python eval_image_enhancement/scripts/plot_v5_bars.py    runs/matrix_v5_full

# 5. Rescue (if any eval container was killed by timeout)
python eval_image_enhancement/scripts/rescue_eval.py runs/matrix_v5_full 8
```

## 6. Results — 100-patient × 10-repeat matrix

See `results/matrix_v5_full/summary.csv` for the full table with bootstrap 95% CIs for pass_rate / agentic_score / clinical_score / overall_score and per-repeat failure breakdown.

### 6.a — Overall / Agentic / Clinical leaderboard (mean over 10 repeats)

| cell                               | agentic | clinical | overall | valid |
|------------------------------------|--------:|---------:|--------:|------:|
| ldct/lite/Claude Opus 4.6          |   1.000 |    0.863 |   0.932 | 1.00  |
| ldct/lite/Gemini 3.1 Pro           |   1.000 |    0.853 |   0.927 | 1.00  |
| ldct/lite/Qwen 3.5                 |   0.864 |    0.845 |   0.855 | 1.00  |
| ldct/lite/MiniMax 2.5              |   0.800 |    0.741 |   0.771 | 1.00  |
| ldct/lite/GPT-5.4                  |   0.693 |    0.756 |   0.725 | 1.00  |
| ldct/lite/GLM-5                    |   0.800 |    0.599 |   0.700 | 0.80  |
| ldct/std/Claude Opus 4.6           |   1.000 |    0.850 |   0.925 | 1.00  |
| ldct/std/Gemini 3.1 Pro            |   1.000 |    0.838 |   0.919 | 1.00  |
| ldct/std/GPT-5.4                   |   1.000 |    0.790 |   0.895 | 1.00  |
| ldct/std/GLM-5                     |   0.934 |    0.804 |   0.869 | 1.00  |
| ldct/std/MiniMax 2.5               |   0.949 |    0.731 |   0.840 | 1.00  |
| ldct/std/Qwen 3.5                  |   0.473 |    0.502 |   0.487 | 0.60  |
| mri-sr/std/GPT-5.4                 |   1.000 |    0.438 |   0.719 | 1.00  |
| mri-sr/std/MiniMax 2.5             |   1.000 |    0.435 |   0.717 | 1.00  |
| mri-sr/std/Claude Opus 4.6         |   0.879 |    0.466 |   0.673 | 1.00  |
| mri-sr/std/Gemini 3.1 Pro          |   0.843 |    0.421 |   0.632 | 0.90  |
| mri-sr/std/Qwen 3.5                |   0.800 |    0.349 |   0.575 | 0.80  |
| mri-sr/std/GLM-5                   |   0.702 |    0.408 |   0.555 | 0.90  |
| mri-sr/lite/Gemini 3.1 Pro         |   0.932 |    0.270 |   0.601 | 1.00  |
| mri-sr/lite/Claude Opus 4.6        |   0.585 |    0.325 |   0.455 | 1.00  |
| mri-sr/lite/GPT-5.4                |   0.536 |    0.303 |   0.420 | 0.90  |
| mri-sr/lite/GLM-5                  |   0.612 |    0.224 |   0.418 | 0.90  |
| mri-sr/lite/MiniMax 2.5            |   0.564 |    0.227 |   0.395 | 0.70  |
| mri-sr/lite/Qwen 3.5               |   0.356 |    0.266 |   0.311 | 1.00  |

A few things this table reveals that pass_rate alone does not:
- **MiniMax's MRI-SR/std overall (0.72) ≈ GPT-5.4's (0.72)** — despite different agentic (1.0 vs 1.0) and clinical (0.44 vs 0.44) paths to get there. Both produced every output and hit the same clinical ceiling.
- **Gemini's MRI-SR/lite agentic=0.93 is misleading alone**: it submitted cleanly (high S4+S5) but its clinical=0.27 drags overall down to 0.60 — still the best MRI-SR/lite cell, which is the signal the old pass_rate-only view missed.
- **Qwen LDCT/std collapses on agentic (0.47)** — not because outputs are bad but because 4/10 runs are `F_no_output`; S4 completeness gate drops the weighted mean.

### 6.b — Pass_rate detail

**pass_rate mean [95% CI]** — *bootstrap mean over the `n_scorable` repeats (out of 10) where the eval container produced scores; `F_no_output` and `F_breach` runs are not in the mean*. Always read alongside `valid_rate = n_scorable / 10`. Example: GPT-5.4 LDCT/lite headline `0.67` is the mean over the 10 scorable runs; but 8 of them have ratings F/F/F/F/B/F/F/F/C/F (many `F_timeout`), which is why the CI is wide.

| agent              | LDCT/lite           | LDCT/std           | MRI-SR/lite         | MRI-SR/std          |
|--------------------|---------------------|--------------------|---------------------|---------------------|
| Claude Opus 4.6    | **0.95** [0.92,0.97] *(valid 1.0)* | 0.90 [0.87,0.92] *(1.0)* | 0.20 [0.06,0.34] *(1.0)* | 0.55 [0.47,0.59] *(1.0)* |
| Gemini 3.1 Pro     | 0.90 [0.90,0.90] *(1.0)* | 0.87 [0.74,0.96] *(1.0)* | 0.03 [0.01,0.07] *(1.0)* | 0.58 [0.55,0.59] *(0.9)* |
| GPT-5.4            | 0.67 [0.51,0.82] *(1.0)* | 0.69 [0.47,0.88] *(1.0)* | 0.34 [0.09,0.59] *(0.9)* | **0.73** [0.72,0.74] *(1.0)* |
| GLM-5              | 0.61 [0.34,0.86] *(0.8)* | 0.78 [0.59,0.93] *(1.0)* | 0.01 [0.01,0.01] *(0.9)* | 0.40 [0.22,0.55] *(0.9)* |
| MiniMax 2.5        | 0.49 [0.30,0.67] *(1.0)* | 0.58 [0.33,0.80] *(1.0)* | 0.29 [0.08,0.55] *(0.7)* | 0.72 [0.69,0.74] *(1.0)* |
| Qwen 3.5           | 0.76 [0.61,0.88] *(1.0)* | 0.69 [0.48,0.89] *(0.6)* | 0.08 [0.00,0.23] *(1.0)* | 0.73 [0.73,0.73] *(0.8)* |

For the full per-repeat breakdown (every run's rating, failure label, PSNR/SSIM/LPIPS, turns, tokens, wall-clock), see `results/matrix_v5_full/summary.csv` and `summary.json`.

**Key observations:**
- **Task-dependent rankings.** Opus/Gemini dominate LDCT but collapse on MRI-SR lite (they strictly execute the Swin2SR mandate, which fails on medical MRI grayscale). GPT-5.4 / MiniMax / Qwen take the opposite shape — mediocre on LDCT, best on MRI-SR std.
- **MRI-SR lite is the discriminator.** Six agents × 10 repeats all cluster near PSNR 19–22 dB / SSIM 0.4 because the constrained DNN (general-image Swin2SR weights) does not transfer to medical MRI. Passing this cell requires ignoring the lite mandate, which most agents do not.
- **Rating vs pass_rate.** Pass_rate is the primary metric. The A band (beat DRUNet by ≥ 1 dB) is basically unreachable with off-the-shelf methods; B/C correctly separate classical from sub-classical.
- **Agent-level failure modes are visible.** `F_breach / F_timeout / F_no_output / F_below_band` are separated in `summary.csv`. GPT-5.4 accumulates `F_timeout` from `execute_code` bash hangs; MiniMax accumulates `F_breach` on prompt confusion; Qwen accumulates `F_no_output` on MRI-SR std.

## 7. Figures

All figures are regenerated by `plot_v5_summary.py` and `plot_v5_bars.py`. Both scripts fall back to the committed `results/matrix_v5_full/` snapshot when the host's `runs/matrix_v5_full/` has been cleaned. Caveat: the **stage-timing figure** (`matrix_v5_full_bars_stage_timing.png`) needs per-run `workspace/process/messages.json` and `tool_log.jsonl` — those are not committed (size), so regenerating that specific figure requires the original `runs/` tree. The already-committed PNG/PDF in `figures/` is the canonical stage-timing plot.

- `figures/matrix_v5_full_pass_rate.png` — 6×4 heatmap with per-repeat rating strings
- `figures/matrix_v5_full_bars_pass_rate.png` — per-task bars, colored by modal rating, annotated with PSNR
- `figures/matrix_v5_full_scatter_{ldct,mri-sr}.png` — PSNR × SSIM with baseline markers + A/B band lines
- `figures/matrix_v5_full_bars_stage_timing.png` — stacked S1–S5 wall-time per cell
- `figures/matrix_v5_full_bars_token_fairness.png` — input tokens per patient, log scale, Lite vs Standard
- `figures/matrix_v5_full_efficiency.png` — wall-time × prompt tokens per repeat

## 8. Intentional contract changes vs `eval_seg`

v5 deliberately extends seg's architecture with:

- **`pre_submit_check` 50% gate** — agent must have ≥ 50% valid `enhanced.npy` before `submit_results` is accepted; otherwise it is rejected and the agent may retry. Seg does not gate submission.
- **Training-pattern sandbox block** — inference-only discipline is enforced statically (`.backward()`, `optimizer.step()`, `model.train()`, `torch.optim.*`).
- **Host UID/GID propagation** — avoids root-owned files. Seg runs as root inside the container.
- **Pinned dependency versions** in both images.
- **Per-turn checkpointing** of `agent_summary.json` / `tool_log.jsonl` / `trace.jsonl` so a killed container still produces audit logs.

These are noted in the CSV (failure_counts) and are not bugs.

## 9. Known limitations

- MRI-SR lite requires general-image Swin2SR; an agent that follows the mandate is penalized for the out-of-domain failure. This is a deliberate discriminator signal, not a bug — standard tier exists to measure unconstrained performance.
- `rating` bands are anchored on classical baselines (BM3D for LDCT, bicubic for MRI-SR). The A band is therefore structurally hard to reach without medical-domain fine-tuned weights.
- Stage timing detection uses regex markers in assistant text; agents that do not write explicit `S1 PLAN` / `S2 SETUP` markers collapse everything into S1 in the stacked plot. This does not affect numeric results.
