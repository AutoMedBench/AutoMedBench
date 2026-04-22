# MedAgentsBench — Phase 1 Plan

## 1. Agent Targets

11 agents across 7 providers:

| Provider   | Models                          | Count |
|------------|---------------------------------|-------|
| OpenAI     | GPT-5.4, o3-pro                 | 2     |
| Anthropic  | Claude Opus 4.6, Sonnet 4.6     | 2     |
| Google     | Gemini 3.1 Pro, Gemini 3 Flash  | 2     |
| Alibaba    | Qwen 3.6 Plus                   | 1     |
| Moonshot   | Kimi K2.5                        | 1     |
| DeepSeek   | V4, R1                          | 2     |
| Z AI       | GLM 5.1                         | 1     |

---

## 2. Tasks

### A. Lesion Segmentation (2 tasks, initial release)

| Task | Organ | Data Source | Metrics |
|------|-------|-------------|---------|
| Task 1 | **Kidney** | CruzBench (8 patients) | Dice (organ + lesion) |
| Task 2 | **Liver** | CruzBench (8 patients) | Dice (organ + lesion) |

> Expandable to pancreas and more organs later.

### B. Medical VQA (3 tasks, planned)

| Dataset | Domain | Task Type |
|---------|--------|-----------|
| PathVQA | Pathology | Slide Q&A + retrieval |
| VQA-RAD | Radiology | Image question answering |
| MedFrameQA | Clinical | Multi-image reasoning |

### C. Image Enhancement (optional)

| Task | Data | Metrics |
|------|------|---------|
| Contrast phase + denoise | JHU/UCSF (private) | SSIM, PSNR |

---

## 3. Success Criteria

- Agents must use **heavy tool calling** (segmentation tools, retrieval, quantification)
- **End-to-end workflow:** input → processing → structured output
- **Traceable summaries** with citations / references

---

## 4. Milestones

### M1 — Infrastructure & Data ✅ (partial)
- [x] Set up eval harness (run_eval.py + scoring modules)
- [x] Stage CruzBench data (kidney + liver, public/private split)
- [x] Create dummy agents for testing (perfect, partial, empty)
- [ ] Define agent interface contract (input schema, tool definitions, output schema)
- [ ] Prepare VQA datasets (PathVQA, VQA-RAD, MedFrameQA)

### M2 — Tool & Workflow Definitions
- [ ] Implement segmentation tools (model inference wrappers agents can call)
- [ ] Implement VQA retrieval tools (slide retrieval, image loaders)
- [ ] Define and document workflow templates for each task type

### M3 — Agent Integration
- [ ] Build provider adapters for each agent family
- [ ] Implement tool-calling translation layer per provider API
- [ ] Validate basic connectivity and tool-use round-trips

### M4 — Benchmark Runs & Reporting
- [ ] Run all agents across all tasks (batch + retry logic)
- [ ] Collect traces, tool-call logs, and structured outputs
- [ ] Generate per-agent and cross-agent comparison report

---

## 5. Benchmark Workflow

```
┌─────────────────────────────────────────────────────────┐
│                  benchmark_runner.py                     │
│                                                         │
│  1. AGENT RUN  (GPU active)                             │
│     ┌─────────────────────────────────────────┐         │
│     │  S1 Plan/Research  (tier-dependent)      │         │
│     │  S2 Setup          (venv + model load)   │         │
│     │  S3 Validate       (1 patient test)      │         │
│     │  S4 Inference      (all patients, GPU)   │         │
│     │  S5 Submit         (CSV + verify)        │         │
│     └─────────────────────────────────────────┘         │
│              ↓ agent finishes, GPU freed                 │
│                                                         │
│  2. DETERMINISTIC EVAL  (CPU only)                      │
│     ┌─────────────────────────────────────────┐         │
│     │  format_checker  → dice_scorer           │         │
│     │  → patient_scorer → medal_tier           │         │
│     │  → aggregate → failure_classifier        │         │
│     └─────────────────────────────────────────┘         │
│              ↓                                           │
│  3. LLM JUDGE  (required, scores S1-S3)                  │
│     ┌─────────────────────────────────────────┐         │
│     │  Loads DeepSeek-R1-Distill-Qwen-32B     │         │
│     │  via vLLM (offline) or Claude (online)   │         │
│     │  Scores S1-S3 + failure analysis         │         │
│     │  Scores S1-S3 (required, no heuristic)    │         │
│     └─────────────────────────────────────────┘         │
│              ↓                                           │
│  4. DETAIL REPORT                                       │
│     ┌─────────────────────────────────────────┐         │
│     │  generate_detail_report()                │         │
│     │  → diagnostic metrics, agentic score,    │         │
│     │    error analysis, step failures,        │         │
│     │    tool call summary (phase breakdown)   │         │
│     └─────────────────────────────────────────┘         │
│              ↓                                           │
│  5. SUMMARY PLOTS  (Pro tier only)                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Key: the judge runs AFTER the agent benchmark finishes and the GPU is
freed from inference. This allows the offline judge (DeepSeek-R1 32B)
to load onto the same GPU without conflict.

---

## 6. Directory Structure

```
MedAgentsBench/
  PLAN.md                   # this file
  eval_seg/                 # segmentation eval pipeline (built)
    run_eval.py             # main entry point
    format_checker.py       # submission format validation
    dice_scorer.py          # organ + lesion Dice
    patient_scorer.py       # patient-level stats (legacy, not scored)
    medal_tier.py           # result tier (good / okay / fail)
    aggregate.py            # composite scoring
    failure_classifier.py   # F1–F5 failure codes
    data/                   # staged data (kidney, liver)
    dummy_agents/           # test submissions
    results/                # eval reports
  configs/                  # agent configs, eval settings (planned)
  tools/                    # tool implementations agents can call (planned)
  agents/                   # provider adapters and agent runners (planned)
  results/                  # cross-agent comparison reports (planned)
```
