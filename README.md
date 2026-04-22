# AutoMedBench

> **Can AI agents autonomously conduct Medical AutoResearch?**

AutoMedBench scores autonomous coding agents across the medical research pipeline — **plan, setup, validate, infer, submit** — not just their final outputs.

*A joint project from UC Santa Cruz × NVIDIA.*

---

## What is AutoMedBench?

AutoMedBench is a benchmark suite for evaluating coding-agent LLMs on realistic medical imaging and reasoning tasks. Each agent must autonomously plan a research approach, set up the environment, validate its pipeline, run inference on held-out patients, and submit results — all inside an isolated sandbox.

The benchmark measures **how** agents work, not just **what** they score. Every run is graded across five stages (S1 Plan → S5 Submit) in addition to the final clinical metric.

## Task domains

Tasks live on separate branches so each domain is independently versioned and reproducible.

| Branch | Domain | Tasks | Status |
|---|---|---|---|
| [`eval_seg`](../../tree/eval_seg) | 3D medical segmentation | kidney · liver · pancreas · feta (fetal brain multi-tissue) | **live** |
| [`eval_image_enhancement`](../../tree/eval_image_enhancement) | 2D image enhancement | LDCT denoising · MRI super-resolution | **live** |
| [`eval_report_gen`](../../tree/eval_report_gen) | CXR report generation | MIMIC-CXR findings | **live** |
| [`eval_vqa`](../../tree/eval_vqa) | Medical VQA | PathVQA · VQA-RAD · SLAKE · MedFrameQA · MedXpertQA-MM | **live** |
| [`eval_det2d`](../../tree/eval_det2d) | 2D detection | VinDr-CXR abnormality | **beta** |

## Quickstart

```bash
# 1. Clone the domain branch you want to benchmark
git clone --branch eval_seg --single-branch https://github.com/KumaKuma2002/AutoMedBench.git
cd AutoMedBench

# 2. Pull the sandbox container (see the branch's README for the exact tag)
docker pull <registry>/automedbench-seg:v1

# 3. Stage the dataset (public inputs + held-out references)
python stage_data.py

# 4. Configure your agent (edit agent_config.yaml)

# 5. Run the benchmark
python docker/orchestrator.py \
    --agent claude-opus-4-6 \
    --task kidney-seg-task \
    --tier lite \
    --n-patients 20
```

Each branch ships its own `README.md` with the exact `docker pull` tag, dataset recipe, and orchestrator invocation.

## Core components

### Task definitions

Each task is a self-contained directory under its domain branch. At minimum, a task provides:

- `config.yaml` — task metadata (modality, patient count, time budget)
- `model_info.yaml` — candidate models for Standard-tier runs
- Skill markdown files (`lite_s1.md` / `standard_s1.md` / …) — tier-specific hints
- `requirements.txt` — Python dependencies pinned for reproducibility

### Execution harness

Every domain uses a two-container architecture:

- **Agent container** — GPU + network, runs the LLM coding loop. Read-only rootfs, sandboxed filesystem.
- **Eval container** — GPU + `--network none`, scores agent outputs against held-out references.

The orchestrator chains them sequentially and applies an isolation-breach penalty if the agent violates sandbox rules.

### Scoring

Every run produces an **Overall** score in `[0, 1]`:

```
Overall = 0.5 × Agentic + 0.5 × Task
```

- **Agentic** — weighted mean of S1–S5 stage scores (see each branch's `SCORING_RUBRICS.md`)
- **Task** — the domain-specific metric (Dice for seg, SSIM for enhancement, clinical score for report, accuracy for VQA, mAP for detection)

## Submit to the leaderboard

Each live-branch README includes a `runs/` layout that matches what the public leaderboard expects. Submission instructions TBD.

## License

Research-use only. Not for clinical deployment.

## Citation

```bibtex
@misc{automedbench2026,
  title  = {AutoMedBench: A Benchmark for Autonomous Medical Research Agents},
  author = {KumaKuma2002 and collaborators},
  year   = {2026},
  url    = {https://github.com/KumaKuma2002/AutoMedBench}
}
```
