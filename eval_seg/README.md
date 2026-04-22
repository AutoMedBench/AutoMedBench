# eval_seg — Segmentation Benchmark

Evaluates LLM agents on medical image segmentation tasks. Agents must autonomously plan, code, and execute organ/lesion segmentation workflows end-to-end.

## Agent Workflow (S1-S5)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  S1 PLAN (25%)        Research models, write plan.md            │
│       ↓                                                         │
│  S2 SETUP (15%)       Download weights, install deps, load GPU  │
│       ↓                                                         │
│  S3 VALIDATE (35%)    Test on 1 patient, check lesion output    │
│       ↓                                                         │
│  S4 INFERENCE (15%)   Run all patients, save masks              │
│       ↓                                                         │
│  S5 SUBMIT (10%)      Verify outputs, call submit_results       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Three tiers control how much help the agent gets:

| | Lite | Standard | Pro |
|---|---|---|---|
| Model info | Exact model given | 2 candidate families | Clinical background only |
| Dependencies | requirements.txt provided | Agent figures out | Agent figures out |
| Skill blocks | S1-S3 | S1 only | None |

## Scoring

```
                          Overall Score
                       (50% Agentic + 50% Clinical)
                      /                              \
            Agentic Score                      Clinical Score
       (weighted S1-S5 avg)              (50% lesion + 50% organ Dice)
      /    |     |     |    \                      |
    S1    S2    S3    S4    S5          lesion: positive pts / organ: all pts
    25%   15%   35%   15%   10%
```

### Step Scores

```
S1 Plan (25%)  — 6 binary sub-criteria, averaged
├── s1a  Created plan.md?                        {0, 1}
├── s1b  Plan has clear pipeline instructions?   {0, 1}  (0 if no plan.md)
├── s1c  Chose lesion model (not general-only)?  {0, 1}  (0 if no plan.md)
├── s1d  Researched >= 3 models?                 {0, 1}  (Lite: always 1)
├── s1e  Created plan visualization (png)?       {0, 1}  (Lite: always 1)
└── s1f  Plot has clear pipeline diagram?        {0, 1}  (0 if no plot; Lite: always 1)
    Score = (s1a + s1b + s1c + s1d + s1e + s1f) / 6

S2 Setup (15%)  — 5 binary sub-criteria, averaged
├── s2a  Checkpoint downloaded?               {0, 1}
├── s2b  Compatibility check done?            {0, 1}
├── s2c  Environment setup succeeded?         {0, 1}
├── s2d  Env failures <= 5 attempts?          {0, 1}
└── s2e  Model loaded on GPU?                 {0, 1}
    Score = (s2a + s2b + s2c + s2d + s2e) / 5

S3 Validate (35%)
├── 0.0  No validation detected
├── 0.2  Basic validation (shape match / voxel count)
├── 0.5  Lesion-aware validation (lesion + tumor in output)
└── 1.0  Single-patient test before batch, with inference output

S4 Inference (15%)  — CONTINUOUS
└── 0.50 * completion_rate + 0.50 * masks_format_valid
    completion_rate = patients_with_output / total_patients  (0.0 to 1.0)
    masks_format_valid = 1.0 if ALL masks are binary + correct shape, else 0.0
    Example: 18/20 patients done, masks valid → 0.50 * 0.9 + 0.50 * 1.0 = 0.95

S5 Submit (10%)  — DISCRETE {0.0, 0.5, 1.0}
└── 0.50 * has_valid_results + 0.50 * output_format_valid
    has_valid_results = 1 if any patient scored AND (lesion_dice > 0 OR organ_dice > 0)
    output_format_valid = 1 if all masks pass format check
    Both are binary (0 or 1), so S5 can only be 0.0, 0.5, or 1.0
```

S1-S3 are scored by the **LLM judge** (required). S4 is **continuous**
(depends on how many patients the agent completed). S5 is **discrete**.

**IMPORTANT: All patients must be completed.** If any patient is missing output,
the entire run is rated F with zero Dice credit. There is no partial credit for
clinical scores — the agent must process every patient or the run fails. S4 still
reflects completion rate for diagnostic purposes, but clinical score and rating
are zeroed on incomplete runs.

### Clinical Score

`= 0.50 * lesion_dice + 0.50 * organ_dice`
  - lesion_dice: positive patients only (those with GT lesion)
  - organ_dice: ALL patients

### Result Tiers

| Tier | Lesion Dice |
|------|-------------|
| Good | >= 0.70 |
| Okay | >= 0.30 |
| Fail | < 0.30 |

### Rating

| Grade | Meaning |
|-------|---------|
| A | Good result (Dice >= 0.70) |
| B | Okay result (0.30 <= Dice < 0.70) |
| C | Below baseline (Dice < 0.30) |
| F | Failed (no valid output OR incomplete — missing patients) |

**Resolved** = Rating A or B (good or okay result)

## Sandbox & Violation Rules

```
Agent can ONLY access:
  /data/public/          CT scans (read-only)
  /workspace/run_<id>/   Own output directory (read-write)

Agent CANNOT access:
  /data/private/         Ground truth masks
  eval_seg/runs/         Other agents' results
  eval_seg/runs_archive/ Archived runs
  eval_seg/*.py          Benchmark source code
  /workspace/            Other concurrent runs
```

**Violation penalty:**
- 1st violation → WARNING (execution blocked, agent continues)
- 2nd violation → KILL (all scores zeroed, rating = F, disqualified)

## How to Run

### Quick Start

Paste into Claude Code, replace the 5 values, go:

```
Read benchmark.skill. Run the benchmark.
Experimenter: kuma
Agent:        kimik2.5new
Task:         kidney
Tier:         lite
Save to:      ./runs/kuma/bench-kimik2.5-kidney-lite
Do NOT touch, kill, or monitor other processes.
Update tracker.md in Save-to dir after every run (see benchmark.skill).
```

The runner auto-appends a `YYMMDD-6hex` run tag per attempt. Runs never collide.

### Available Tasks

| Task ID | Organ | Modality | Patients | SOTA Model | SOTA Dice |
|---------|-------|----------|----------|------------|-----------|
| kidney-seg-task | kidney | CT | 20 | nnU-Net KiTS19 | 0.85 |
| liver-seg-task | liver | CT | 40 | nnU-Net Task029_LiTS | 0.74 |
| pancreas-seg-task | pancreas | CT | 20 | MedFormer PanTS | 0.52 |

Tasks are auto-discovered from `<task-id>/` folders. Each folder
contains `config.yaml`, `model_info.yaml`, skill files, and `requirements.txt`.

### Adding a New Segmentation Task

Create a folder `<organ>-seg-task/` with:

```
<organ>-seg-task/
  config.yaml          # organ, modality, input_filename, time_limit, etc.
  model_info.yaml      # lite/standard/pro model guidance
  requirements.txt     # dependencies (lite tier)
  lite_s1.md           # S1 skill for lite
  lite_s2.md           # S2 skill for lite
  lite_s3.md           # S3 skill for lite/standard
  standard_s1.md       # S1 skill for standard
  standard_s3.md       # S3 skill for standard
```

No Python changes needed — auto-discovery via `task_loader.py` handles it.

### Output

```
./runs/<experimenter>/bench-<shortname>-<task>-<tier>/<YYMMDD-6hex>/
  detail_report.json    # All scores and metrics
  run.log               # Full stdout/stderr
  process/              # Conversation, trace, tool calls
  outputs/              # Masks, plan

Example: ./runs/kuma/bench-opus-kidney-standard/260412-a3f82b/
```
