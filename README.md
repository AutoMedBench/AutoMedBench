# AutoMedBench

[![Website](https://img.shields.io/badge/Website-automedbench-76B900?style=for-the-badge)](https://keen-bonbon-690c3c.netlify.app/)
[![License](https://img.shields.io/badge/License-Research_only-2B2B25?style=for-the-badge)](LICENSE)

> Towards *Medical AutoResearch* <br>
> — a benchmark for AI agents on medical AI tasks.

---

## 1. Introduction

**AutoMedBench** benchmarks autonomous coding agents on real medical imaging and reasoning tasks — to test how far they can go, unassisted, from "read the clinical problem" to "submit results."

Unlike output-only benchmarks, AutoMedBench grades the *working process* itself. Every run is scored across five stages (**S1 Plan · S2 Setup · S3 Validate · S4 Inference · S5 Submit**) with a strict rubric, not just its final metric:

```
Overall = 0.5 × Agentic (S1-S5 rubric) + 0.5 × Task (clinical metric)
```

---

## 2. Quick Start

AutoMedBench is organized as one branch per task domain. Pick a domain and pull it.

```bash
# 1. Clone the domain branch you want to run
git clone --branch eval_seg --single-branch \
    https://github.com/KumaKuma2002/AutoMedBench.git
cd AutoMedBench

# 2. Pull the sandbox container (tag is in the branch README)
docker pull <registry>/automedbench-seg:v1

# 3. Stage public inputs + private references
python stage_data.py

# 4. Configure your agent (edit eval_seg/agent_config.yaml)

# 5. Run one cell
python eval_seg/docker/orchestrator.py \
    --agent claude-opus-4-6 \
    --task kidney-seg-task \
    --tier lite \
    --n-patients 20
```

Each domain branch ships its own `README.md` with the exact Docker tag, dataset recipe, and runner flags.

---

## 3. Workflow

Every task in every domain runs the same five-stage pipeline. Each stage is scored independently.

```
  ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐
  │ S1 PLAN │ ─▶ │ S2 SETUP │ ─▶ │ S3 VALIDATE │ ─▶ │ S4 INFERENCE │ ─▶ │ S5 SUBMIT │
  └─────────┘    └──────────┘    └─────────────┘    └──────────────┘    └───────────┘
    research       download          pilot one         run the full         verify,
    + plan.md      + install         sample, check      cohort, save         call
                   + GPU load        outputs            per-sample           submit_results
```

- **S1-S3** are scored by an LLM judge against a binary rubric derived from the agent's `plan.md` and tool-call history.
- **S4-S5** are scored deterministically from output completeness and format validity.
- Violating the sandbox (writing to `/data/private/`, reading held-out references) zeros every stage.

Full per-stage rubric: see each branch's `SCORING_RUBRICS.md`.

---

## 4. Further Documentation

- **[Task Gallery](docs/task-gallery.md)** — every task, metric, and source branch at a glance
- **[Dataset Collection](docs/dataset-collection.md)** — which datasets, why, and the public/private split
- **[Task Difficulty Tiers](docs/task-difficulty-tiers.md)** — Lite / Standard / Pro definitions and what each measures

---

## 5. Live Dashboard

A live leaderboard is maintained at **[keen-bonbon-690c3c.netlify.app](https://keen-bonbon-690c3c.netlify.app/)**.

The dashboard shows:
- **Overall Score** across all tasks, per agent
- **S1-S5 per-stage breakdown** — you can see *where* each agent fails, not just whether it failed
- **Per-task leaderboards** with Dice / SSIM / accuracy / mAP distributions
- **Cost, turns, wall-time, token use** per run

Currently live: segmentation + image enhancement. VQA and report generation are scored pending.

---

## 6. Core Components

### Task definitions

Each task is a self-contained directory under its domain branch:

```
eval_seg/kidney-seg-task/
  config.yaml           # task metadata (modality, patient count, time budget)
  model_info.yaml       # candidate models for Standard-tier runs
  requirements.txt      # pinned Python dependencies
  lite_s1.md            # tier-specific skill hints for S1
  lite_s2.md            ...
  standard_s1.md        ...
```

### Execution harness

Two-container architecture per domain:

- **Agent container** — GPU + network, runs the LLM coding loop. Read-only rootfs, 3-layer filesystem sandbox.
- **Eval container** — GPU + `--network none`, scores agent outputs against held-out references.

The orchestrator (`docker/orchestrator.py`) chains them sequentially and applies an isolation-breach penalty if the agent violates sandbox rules.

### Scoring

```
Overall = 0.5 × Agentic + 0.5 × Task
Agentic = weighted mean of S1-S5 stage scores
          (S1 25% · S2 15% · S3 35% · S4 15% · S5 10%)
Task    = mean Dice / SSIM / clinical / accuracy / mAP
```

See `SCORING_RUBRICS.md` on each branch for the full per-sub-criterion breakdown.

---

## 7. Contribution

### Adding a task

1. Pick the domain branch that matches your task.
2. Copy an existing task directory (e.g. `eval_seg/kidney-seg-task/`) as a template.
3. Fill in `config.yaml`, `model_info.yaml`, tier-specific skill markdown files.
4. Stage a dataset that matches the three requirements above.
5. Open a pull request.

### Adding a new domain

Create a new `eval_<domain>/` top-level directory on its own branch, mirroring the structure of `eval_seg/`. See the segmentation branch for a reference implementation — the S1-S5 harness, LLM judge, and Docker orchestrator are all reusable.

### Submitting to the leaderboard

Produce the standard `runs/` layout (a `detail_report.json` + `workspace/process/conversation.json` per repeat). Submission instructions and the evaluation server are in flight — watch this README for updates.

---

<p align="center">
  <a href="https://www.ucsc.edu/"><img src="assets/ucsc-logo.svg" alt="UC Santa Cruz" height="32"></a>
  &nbsp;&nbsp;×&nbsp;&nbsp;
  <a href="https://www.nvidia.com/"><img src="assets/nvidia-logo.svg" alt="NVIDIA" height="32"></a>
</p>
