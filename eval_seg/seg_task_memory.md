# MedAgentsBench — Project Memory

## What This Is

A benchmark framework that evaluates LLM agents on medical image segmentation tasks.
Agents must autonomously plan, set up, and execute organ + lesion segmentation pipelines
on CT scans from the CruzBench dataset.

## Key Rules

1. **NEVER modify files outside `kuma_workspace/`**
2. **NEVER touch the root folder (`/root/` or `/`)** — it will crash the system
3. All work stays within this directory tree

## Directory Layout

```
MedAgentsBench/
  PLAN.md                     # Phase 1 roadmap (11 agents, 7 providers)
  project_memory.md           # This file
  eval_seg/
    agent_config.yaml          # All agent definitions + pricing + settings
    benchmark_runner.py        # REAL benchmark — agent gets execute_code only (S1-S5)
    agent_runner.py            # Tool-calling mode — agent gets 7 pre-defined tools
    agent_prompt.py            # System prompts + tool schemas for tool-calling mode
    agent_tools.py             # SegmentationTools class (dispatch, mock, stats)
    run_eval.py                # Evaluation entry point (format + dice + patient + medal + aggregate)
    aggregate.py               # S4/S5 scoring, workflow/clinical/overall score, rating (A/B/C/F)
    dice_scorer.py             # Organ + lesion Dice computation
    patient_scorer.py          # Patient-level stats (legacy, not scored)
    medal_tier.py              # Result tier assignment (good / okay / fail)
    format_checker.py          # Submission format validation
    failure_classifier.py      # E1-E5 error codes
    detail_report.py           # Generates the final detail_report.json
    llm_judge.py               # LLM-as-judge (online: Claude Opus 4.6, offline: DeepSeek-R1)
    stage_data.py              # Data staging script
    make_dummy_agents.py       # Creates test submissions (perfect, partial, empty)
    data/                      # Staged data (kidney/, liver/) — public + private splits
    dummy_agents/              # Test submissions for validation
    bundles/                   # MONAI bundles (vista3d, wholeBody_ct_segmentation)
    runs/                      # All benchmark run outputs (timestamped)
  models/                      # Model weights
  pip_packages/                # Local pip packages (supervisor, mpmath, etc.)
```

## Two Runner Modes

### 1. Tool-Calling Mode (`agent_runner.py`)
- Agent gets 7 tools: list_patients, get_patient_ct, run_organ_segmentation,
  run_lesion_detection, read_mask_stats, submit_decision, finalize
- Model inference handled by framework; agent orchestrates
- `python agent_runner.py --agent claude-opus-4-6 --task kidney [--mock]`

### 2. Real Benchmark Mode (`benchmark_runner.py`) — 3 Tiers
- Agent gets ONE tool: `execute_code` (python or bash in conda env)
- Agent must code its own pipeline following S1-S5 workflow
- All tiers: S1 PLAN → S2 SETUP → S3 VALIDATE → S4 INFERENCE → S5 SUBMIT
- Same scoring system, same patients, same limits across all tiers
- Isolation enforced: blocked from private data, ground truth, dummy agents
- Runs in `runs/<tier>/<agent>/<task>/<timestamp>/`

#### Tier Differences (instructions only, not scoring)

| Feature | Lite | Standard | Pro |
|---------|------|----------|-----|
| S1 Model | SOTA architecture given, agent web-searches for checkpoint | Model family range given | Nothing given, compete |
| S1 Skill | Yes | Yes | No |
| S2 Env | venv + provided requirements.txt (+ may install more) | venv + agent figures deps | venv + agent figures deps |
| S2 Skill | Yes | No | No |
| S3 Skill | Yes | No | No |
| S4 Post-proc | Disabled in prompt | Disabled in prompt | Enabled + research |
| plan.md | Yes | Yes | Yes |
| plan.png | No | Yes | Yes |
| Summary Plots | No | No | Yes (cozy minimalism) |

Usage:
  `python benchmark_runner.py --agent claude-opus-4-6 --task kidney --tier lite`
  `python benchmark_runner.py --agent claude-opus-4-6 --task kidney --tier standard`
  `python benchmark_runner.py --agent claude-opus-4-6 --task kidney --tier pro`
- `python benchmark_runner.py --agent claude-opus-4-6 --task kidney`

## Scoring System

- **Step scores**: S1 (plan, 25%), S2 (setup, 15%), S3 (validate, 35%), S4 (inference, 15%), S5 (submit, 10%)
- **S1-S3**: scored by LLM judge (required), sub-criteria averaged
- **S4-S5**: deterministic (completion rate + format check)
- **Agentic score**: weighted avg of S1-S5
- **Clinical score**: 50% lesion_dice (positive pts) + 50% organ_dice (all pts)
- **Overall score**: 50% agentic + 50% clinical
- **Rating**: A (good, Dice >= 0.70), B (okay, Dice >= 0.30), C (below baseline), F (failed/incomplete)
- **Resolved**: Rating A or B (good or okay result)

## Agents Configured

11 agents across 7 providers, all defined in `agent_config.yaml`:
- OpenAI: GPT-5.4, o3-pro
- Anthropic (via OpenRouter): Claude Opus 4.6, Sonnet 4.6
- Google: Gemini 3.1 Pro, Gemini 3 Flash
- Alibaba: Qwen 3.6 Plus
- Moonshot: Kimi K2.5
- DeepSeek: V4, R1
- Z AI: GLM 5.1

## Tasks

- **Kidney**: 8 CruzBench patients (BDMAP_00000001..862) — active
- **Liver**: 8 CruzBench patients — configured but patient IDs not populated yet

## Latest Run (2026-04-10)

**Run**: `runs/claude-opus-4-6_benchmark/kidney/20260410_133512/`
- Mode: Real benchmark (execute_code only)
- Overall: 0.5806, Rating C, NOT resolved
- Organ dice: 0.903 (good), Lesion dice: 0.071 (poor)
- Lesion detection was poor (wrong model choice)
- 34 API calls, 52 code executions, ~32 min, $20.70
- **Root cause**: Agent chose TotalSegmentator (kidney_cyst task) which only detects
  cysts, not tumors. It identified KiTS21 nnU-Net as the better model in S1 research
  but chose the easier option. Result: missed real tumors (FN) and false-flagged
  non-lesion patients (FP).

## Run History

```
runs/claude-opus-4-6_benchmark/kidney/
  20260410_133512/  ← latest real benchmark (detail above)
  20260410_125645/
  20260410_125645/
  20260409_224125/
  20260409_222147/
  20260409_220617/
  20260409_220137/
  20260409_220119/

runs/claude-opus-4-6/kidney/        ← tool-calling mode runs
  20260409_215324/
  20260409_215058/
  20260409_210848/
  20260409_195743/
  20260409_195737/

runs/claude-opus-4-6/liver/
  20260409_200021/
```
