# AutoMedBench

[![Website](https://img.shields.io/badge/Website-automedbench-76B900?style=for-the-badge)](https://keen-bonbon-690c3c.netlify.app/)
[![License](https://img.shields.io/badge/License-Research_only-2B2B25?style=for-the-badge)](LICENSE)

> Towards *Medical AutoResearch* <br>
> — a benchmark for AI agents on medical AI tasks.

AutoMedBench scores autonomous coding agents across the medical research pipeline — **plan, setup, validate, infer, submit** — not just their final outputs.

```
  ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐
  │ S1 PLAN │ ─▶ │ S2 SETUP │ ─▶ │ S3 VALIDATE │ ─▶ │ S4 INFERENCE │ ─▶ │ S5 SUBMIT │
  └─────────┘    └──────────┘    └─────────────┘    └──────────────┘    └───────────┘
    research       download          pilot one         run the full         verify,
    + plan.md      + install         sample, check      cohort, save         call
                   + GPU load        outputs            per-sample           submit_results
```

Every task in every domain runs through the same five stages. Each is scored independently.

---

## Quickstart

```bash
# 1. Clone the domain branch you want to run
git clone --branch eval_seg --single-branch https://github.com/KumaKuma2002/AutoMedBench.git
cd AutoMedBench

# 2. Pull the sandbox container (tag is in the branch README)
docker pull <registry>/automedbench-seg:v1

# 3. Stage the public + private splits
python stage_data.py

# 4. Run one cell
python eval_seg/docker/orchestrator.py \
    --agent claude-opus-4-6 \
    --task kidney-seg-task \
    --tier lite
```

Each domain branch ships its own `README.md` with the exact Docker tag, dataset recipe, and runner flags.

## Domains

| Branch | Domain | Tasks | Status |
|---|---|---|---|
| [`eval_seg`](../../tree/eval_seg) | 3D segmentation | kidney · liver · pancreas · feta | live |
| [`eval_image_enhancement`](../../tree/eval_image_enhancement) | image enhancement | LDCT · MRI-SR | live |
| [`eval_report_gen`](../../tree/eval_report_gen) | CXR report generation | MIMIC-CXR | live |
| [`eval_vqa`](../../tree/eval_vqa) | medical VQA | PathVQA · VQA-RAD · SLAKE · MedFrameQA · MedXpertQA-MM | live |
| [`eval_det2d`](../../tree/eval_det2d) | 2D detection | VinDr-CXR | beta |

## Scoring

```
Overall = 0.5 × Agentic + 0.5 × Task
```

- **Agentic** — weighted mean of S1–S5 stage scores (Plan · Setup · Validate · Infer · Submit)
- **Task** — domain-specific metric (Dice · SSIM · clinical score · accuracy · mAP)

---

<p align="center">
  <a href="https://www.ucsc.edu/"><img src="assets/ucsc-logo.svg" alt="UC Santa Cruz" height="32"></a>
  &nbsp;&nbsp;×&nbsp;&nbsp;
  <a href="https://www.nvidia.com/"><img src="assets/nvidia-logo.svg" alt="NVIDIA" height="32"></a>
</p>
