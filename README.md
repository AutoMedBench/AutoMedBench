# AutoMedBench

[![Website](https://img.shields.io/badge/Website-automedbench-76B900?style=for-the-badge)](https://keen-bonbon-690c3c.netlify.app/)
[![Sandbox](https://img.shields.io/badge/Sandbox-coming_soon-FFD21E?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-Research_only-2B2B25?style=for-the-badge)](LICENSE)

> Towards *Medical AutoResearch* <br>
> — a benchmark for AI agents on medical AI tasks.

<p align="center">
  <img src="assets/workflow.png" alt="AutoMedBench agent workflow — S1 Plan, S2 Setup, S3 Validate, S4 Inference, S5 Submit" width="960">
</p>

---

## 1. Introduction

**AutoMedBench** benchmarks autonomous coding agents on real medical imaging and reasoning tasks — to test how far they can go, unassisted, from "read the medical research problem" to "submit results."

Unlike output-only benchmarks, AutoMedBench grades the *working process* itself. Every run is scored across five stages (**S1 Plan · S2 Setup · S3 Validate · S4 Inference · S5 Submit**) with a strict rubric, not just its final metric:

```
Overall = 0.5 × Agentic (S1-S5 rubric) + 0.5 × Task (task metric)
```

---

## 2. Quick Start

Sandbox containers and datasets are hosted on HuggingFace.

<p align="left">
  <a href="#"><img src="https://img.shields.io/badge/Sandbox-coming_soon-FFD21E?style=for-the-badge" alt="Sandbox — coming soon"></a>
</p>

```bash
# 1. Clone the domain branch you want to test
git clone --branch eval_seg --single-branch \
    https://github.com/KumaKuma2002/AutoMedBench.git
cd AutoMedBench

# 2. Pull the sandbox (tag is in each branch README)
docker pull <registry>/automedbench-seg:latest

# 3. Run one task cell
python eval_seg/docker/orchestrator.py \
    --agent claude-opus-4-6 \
    --task kidney-seg-task \
    --tier lite
```

See each branch's `README.md` for the full setup and flags.

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
Task    = mean Dice / SSIM / MLRG-7 / accuracy / mAP
```

See `SCORING_RUBRICS.md` on each branch for the full per-sub-criterion breakdown.

---

## 7. Contribution

We welcome contributions from clinicians, researchers, and engineers — you do not need to know our harness internals to help.

### Have a task idea?
Open a GitHub issue describing a medical problem you'd like agents to tackle: what the input looks like, what the ground truth is, and what "done" means. We'll handle the wiring.

### Want a new domain?
Segmentation, VQA, and report generation are just the start. Propose any medical AI task with deterministic ground truth — we'd love to add it.

### Ran the benchmark on a new agent?
Share your results and we'll get them onto the live dashboard. Submission flow is in flight — open an issue to coordinate.

---

<p align="center">
  <a href="https://www.ucsc.edu/"><img src="assets/ucsc-logo.svg" alt="UC Santa Cruz" height="32"></a>
  &nbsp;&nbsp;×&nbsp;&nbsp;
  <a href="https://www.nvidia.com/"><img src="assets/nvidia-logo.svg" alt="NVIDIA" height="32"></a>
</p>
