# AutoMedBench — Complete UI Specification v2

> **Design Direction**: "Neural Scan" — clinical precision meets AI agent architecture.
> Dark reading-room atmosphere with contrast-dye accent colors.

---

## Part 0: What UIs Does This Project Actually Need?

AutoMedBench is not just a leaderboard. It's a research platform with:

| Asset | What it is | Who needs to see it |
|-------|-----------|---------------------|
| **Harness** | Two-container Docker architecture, isolation checks, entrypoint scripts | Maintainers debugging runs, new contributors |
| **Skills** | Tier-specific hint files (lite_s1.md, standard_s1.md...) — the "curriculum" that guides agents | Researchers comparing tiers, designing new tasks |
| **Rubrics** | Scoring logic in code (aggregate.py, medal_tier.py) — weights, thresholds, rating tiers | Anyone interpreting scores |
| **Runs** | 3,700+ runs with detail_report.json, conversation.json, tool_call logs | Debugging failures, analyzing agent behavior |
| **Tasks** | 58+ task dirs across 5 domains, each with config.yaml, model_info.yaml | Managing the benchmark |
| **Scores** | Summary CSVs, per-task rankings, agent profiles | Public leaderboard, research analysis |

This maps to **two UIs**:

### A. Public Site — automedbench.github.io
For the world: leaderboard, task gallery, agent profiles. The public face of the benchmark.

### B. Ops Console — Internal Dashboard
For the team: harness viewer, skill browser, rubric explorer, run inspector, score matrix. The research control panel.

---

## Part A: Public Site

### A1. Landing Page `/`
- Hero: animated S1→S5 pipeline, wordmark, stats counter
- Domain cards (5 + "Classification coming soon")
- Leaderboard top-5 preview
- Citation block with copy button

### A2. Leaderboard `/leaderboard`
- Global rankings table: Rank | Agent | Overall | Agentic S1-S5 | Task Score | Domains | Cost | Time
- Domain tabs to filter
- Score bars in cells
- Agent-vs-task scatter plot (Agentic score × Task score, quadrant labels)
- Tier selector (Lite / Standard / Pro)

### A3. Domain Leaderboard `/leaderboard/:domain`
- Domain-specific hero with subtle medical image backdrop
- Per-task rankings table within that domain
- Domain metric explainer (Dice / SSIM / mAP / etc.)

### A4. Task Detail `/tasks/:taskId`
- **S1-S5 Pipeline** — signature visualization. Five nodes in a horizontal scan path. Each node shows stage score + weight. Agent selector dropdown to switch whose scores you see.
- Results table: all agents ranked by overall score on this task
- Score distribution histogram
- Task metadata: domain, modality, organ, source, config preview

### A5. Agent Profile `/agents/:agentId`
- Hero: overall score + global rank + model family badge
- 5-axis domain radar chart
- S1-S5 stage profile (horizontal bars — avg per stage)
- Best/worst tasks
- Recent runs table

### A6. Agent Comparison `/compare`
- Multi-select agents (2-4)
- Radar overlay, S1-S5 side-by-side bars
- Task-by-task win/loss table

---

## Part B: Ops Console — Internal Dashboard

This is where the real research happens. The Ops Console surfaces the harness, skills, rubrics, and run data that the public site doesn't need.

### B1. Harness Viewer `/ops/harness`

**What it shows**: The two-container Docker architecture rendered as an interactive diagram.

```
┌─────────────────────────────────────────────────────────┐
│                    HARNESS ARCHITECTURE                   │
│                                                          │
│  ┌──────────────┐         ┌──────────────┐              │
│  │ AGENT CONT.  │ ──────▶ │ EVAL CONT.   │              │
│  │ GPU + net    │ outputs │ --network    │              │
│  │ read-only /  │────────▶│ none         │              │
│  │ agent_loop   │         │ run_eval.py  │              │
│  └──────┬───────┘         └──────┬───────┘              │
│         │ mounts                 │ mounts                │
│    ┌────┴────┐             ┌─────┴─────┐                │
│    │ /data/  │             │ /private/ │                │
│    │ public  │             │ ground    │                │
│    │ (ro)    │             │ truth     │                │
│    └─────────┘             └───────────┘                │
│                                                          │
│  ISOLATION: read-only rootfs | /tmp tmpfs | audit hook  │
│  BREACH → exit 99 → all scores zeroed → Rating F        │
└─────────────────────────────────────────────────────────┘
```

**Interactive elements:**
- Click container → see Dockerfile, entrypoint script
- Click mount point → see what data lives there
- Click "isolation" → expand to show all sandbox rules
- Domain selector: switches between seg/det2d/enhancement/report/vqa architectures
- Tier selector: shows which containers/scripts change per tier

**Data source**: Dockerfiles, entrypoint scripts, orchestrator.py

---

### B2. Skill Browser `/ops/skills`

**What it shows**: The tier-specific skill hint files — compared side-by-side.

```
┌──────────────────────────────────────────────────────────┐
│  SKILL BROWSER                          task: kidney ▼   │
│                                                          │
│  ┌──────────┬──────────┬──────────┐                     │
│  │  LITE    │ STANDARD │   PRO    │  ← tier tabs         │
│  └──────────┴──────────┴──────────┘                     │
│                                                          │
│  ┌──────────┬──────────┬──────────┐                     │
│  │   S1     │   S2     │   S3     │  ← stage tabs        │
│  │  PLAN    │  SETUP   │VALIDATE  │                      │
│  └──────────┴──────────┴──────────┘                     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ # lite_s1.md — Kidney Segmentation               │   │
│  │                                                  │   │
│  │ Skill — How to download and inspect the model:   │   │
│  │ ```python                                        │   │
│  │ from huggingface_hub import snapshot_download    │   │
│  │ model_dir = snapshot_download(                   │   │
│  │   "KagglingFace/nnUNet-KiTS19-3d-lowres-50...    │   │
│  │   local_dir="{output_dir}/model/nnunet_kits19")  │   │
│  │ ```                                              │   │
│  │                                                  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ DIFF VIEW ──────────────────────────────────────┐   │
│  │  lite_s1.md  vs  standard_s1.md                  │   │
│  │                                                  │   │
│  │  Lite: model is HARDCODED (nnUNet KiTS19)        │   │
│  │  Standard: agent SEARCHES HuggingFace, compares  │   │
│  │  + builds comparison table in plan.md            │   │
│  │                                                  │   │
│  │  [Side-by-side code diff]                        │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Core functionality:**
- Task selector dropdown (all 58+ tasks)
- Tier tabs: Lite | Standard | Pro
- Stage tabs: S1 | S2 | S3 (S4/S5 rarely have tier-specific skills)
- Rendered markdown with syntax-highlighted code blocks
- **Diff mode**: Compare lite vs standard, or standard vs pro side-by-side
- Template variable highlighting: `{data_dir}`, `{output_dir}`, `{organ}` shown as highlighted tokens
- "View all skills for this task" — shows all 3-5 files at once in a scrollable panel

**Data source**: `lite_s1.md`, `lite_s2.md`, `lite_s3.md`, `standard_s1.md`, `standard_s3.md` from each task directory

---

### B3. Rubric Explorer `/ops/rubrics`

**What it shows**: The scoring logic visualized as a decision tree — not just raw code.

```
┌──────────────────────────────────────────────────────────┐
│  RUBRIC EXPLORER                   domain: seg ▼         │
│                                                          │
│  OVERALL = 0.50 × AGENTIC + 0.50 × CLINICAL             │
│                                                          │
│  ┌─ AGENTIC SCORE (0.50) ───────────────────────────┐   │
│  │                                                   │   │
│  │  S1 (25%)    S2 (15%)    S3 (35%)   S4 (15%) S5  │   │
│  │  [====  ]    [======]    [=======]  [===   ] [==] │   │
│  │   0.67         1.00        1.00      0.00    0.00 │   │
│  │                                                   │   │
│  │  S4 = 0.50 × inference_completes                  │   │
│  │     + 0.50 × output_format_valid                  │   │
│  │                                                   │   │
│  │  S5 = 0.50 × has_valid_results                    │   │
│  │     + 0.50 × output_format_valid                  │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ CLINICAL SCORE (0.50) ──────────────────────────┐   │
│  │  Binary tasks: 0.50 × lesion_dice                │   │
│  │               + 0.50 × organ_dice                │   │
│  │  Multiclass:   macro_mean_dice                   │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ RATING TIERS ───────────────────────────────────┐   │
│  │  Medal ≥ 2 → A    Medal ≥ 1 → B                  │   │
│  │  Below baseline → C    Invalid format → F         │   │
│  │  Isolation breach → F (ALL scores zeroed)         │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ ISOLATION PENALTIES ────────────────────────────┐   │
│  │  Writing to /data/private/     → breach, Rating F│   │
│  │  Reading held-out references   → breach, Rating F│   │
│  │  Network access in eval        → breach, Rating F│   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Core functionality:**
- Domain selector (seg/det2d/enhancement/report/vqa — each has different scoring)
- Visual weight tree: S1-S5 weights shown as horizontal bars proportional to weight
- Expandable sub-criteria: click S3 → see all sub-checks
- **"What If" calculator**: Drag sliders on S1-S5 scores → see overall score update live
- Threshold visualization: show where A/B/C/F boundaries fall on the score axis
- Link to source code: "View in aggregate.py:22"

**Data source**: `aggregate.py`, `medal_tier.py`, `failure_classifier.py` from each domain

---

### B4. Run Inspector `/ops/runs/:runId`

**What it shows**: Deep drill into a single benchmark run. The debugging tool.

```
┌──────────────────────────────────────────────────────────┐
│  RUN INSPECTOR                                           │
│  bench-gpt5.4-kidney-standard / 260425-986e2e            │
│                                                          │
│  ┌─ HEADER ─────────────────────────────────────────┐   │
│  │  Agent: gpt-5.4    Task: kidney-seg    Tier: std  │   │
│  │  Rating: F    Overall: 0.3337    Wall: 3600s      │   │
│  │  Cost: $2.55    15 API calls    4,614 tokens      │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ S1-S5 TIMELINE ─────────────────────────────────┐   │
│  │                                                   │   │
│  │  ●S1━━━━●S2━━●S3━━━━━━━━━━━━━━━━━━━━━━●S4──●S5   │   │
│  │  0     500   800                   3400  3550 3600│   │
│  │  PLAN  SETUP VALIDATE (stuck 43min!)  INFER SUBMIT│   │
│  │  0.67   1.00  1.00                   0.00  0.00  │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ CONVERSATION ───────────────────────────────────┐   │
│  │  ┌─ Turn 1 (S1) ─────────────────────────────┐   │   │
│  │  │  System: "You are a medical AI agent..."   │   │   │
│  │  │  Agent:  "I'll plan the kidney seg..."     │   │   │
│  │  │  Tool:   execute_code(bash, "pip install") │   │   │
│  │  │  Result: "Successfully installed nnunet"   │   │   │
│  │  └────────────────────────────────────────────┘   │   │
│  │  ┌─ Turn 2 (S1) ─────────────────────────────┐   │   │
│  │  │  Agent:  "Let me download the model..."    │   │   │
│  │  │  Tool:   execute_code(python, "...")       │   │   │
│  │  │  Result: exit_code=0, exec_time=12.3s      │   │   │
│  │  └────────────────────────────────────────────┘   │   │
│  │  ... (13 more turns)                               │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ TOOL CALL LOG ──────────────────────────────────┐   │
│  │  #  Turn  Phase   Language  Exit  Time    Desc    │   │
│  │  1   1     S1      bash      0    0.9s    install │   │
│  │  2   1     S1      python    0    12.3s   download│   │
│  │  ...                                              │   │
│  │  Errors: 1 (E5: inference_run_failure)            │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ ERROR ANALYSIS ─────────────────────────────────┐   │
│  │  Primary failure: E5 — inference_run_failure     │   │
│  │  S4 capped at 0.0 (no valid outputs)             │   │
│  │  S5 capped at 0.0 (nothing to submit)            │   │
│  │  Root cause: agent exhausted time budget in S3   │   │
│  │  trying to validate on patient CT-0012           │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Core functionality:**
- **Timeline**: Horizontal bar showing wall-clock time per stage. Color-coded by stage. Red zone = timeout. Hover = exact timestamps.
- **Conversation viewer**: Scrollable chat log. System prompt (collapsible), agent messages, tool calls with results. Syntax-highlighted code blocks. Search within conversation.
- **Tool call log table**: Sortable (by time, phase, exit code). Filter by error. Click row → jump to conversation.
- **Error analysis panel**: Auto-generated from `error_analysis` and `step_failures` in detail_report.json. Shows root cause and cascading effects.
- **Output browser**: If outputs exist, show sample images (segmentation overlay, enhanced image, etc.), file listing.
- **Raw JSON**: "View detail_report.json" toggle for full machine-readable data.

**Data source**: `detail_report.json` + `conversation.json` + `run.log` from each run directory

---

### B5. Score Matrix `/ops/matrix`

**What it shows**: The big picture — all agents × all tasks × all tiers in a heatmap.

```
┌──────────────────────────────────────────────────────────┐
│  SCORE MATRIX              tier: standard ▼              │
│                                                          │
│           │ colon│kidney│liver│heart│spleen│prostate│... │
│  ─────────┼──────┼──────┼──────┼──────┼──────┼────────┼─ │
│  gpt-5.4  │ 0.62 │ 0.33 │ 0.58 │ 0.71 │ 0.45 │  0.52  │  │
│  claude-o │ 0.74 │ 0.61 │ 0.69 │ 0.83 │ 0.57 │  0.66  │  │
│  gemini31 │ 0.55 │ 0.42 │ 0.51 │ 0.68 │ 0.39 │  0.48  │  │
│  qwen3.5  │ 0.48 │ 0.37 │ 0.44 │ 0.59 │ 0.32 │  0.41  │  │
│  ...      │      │      │      │      │      │        │  │
│                                                          │
│  CELL COLOR: Green (0.7+) → Yellow (0.5+) → Red (<0.3) │
│  CELL TEXT:  Overall score (Agentic / Task)             │
│                                                          │
│  ┌─ FILTERS ────────────────────────────────────────┐   │
│  │  Domain: [x] Seg  [x] Det  [ ] Enh  [ ] Rep ... │   │
│  │  Metric:  [Overall ▼]  Tier: [Standard ▼]        │   │
│  │  Sort by: [Agent name ▼]                          │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  Click any cell → opens Run Inspector for best run       │
│  Click agent name → opens Agent Profile                 │
│  Click task name → opens Task Detail                    │
└──────────────────────────────────────────────────────────┘
```

**Core functionality:**
- Heatmap: rows = agents, columns = tasks. Cell color = score (green → red).
- Cell shows two numbers: overall score + (agentic / task) breakdown.
- Hover: detail popup with full S1-S5 breakdown, rating, cost, wall time.
- Click cell: opens best run's Run Inspector.
- Filters: domain checkboxes, metric selector (overall/agentic/task), tier selector.
- Sort: by agent name, by average score, by domain.
- **Coverage mode**: toggle to show which task×agent combos have been run (green = done, red = missing).
- Export CSV.

**Data source**: Summary CSVs + detail_report.json files

---

### B6. Task Manager `/ops/tasks`

**What it shows**: All 58+ tasks at a glance with configuration details.

```
┌──────────────────────────────────────────────────────────┐
│  TASK MANAGER                                            │
│                                                          │
│  ┌─ FILTERS ────────────────────────────────────────┐   │
│  │  Domain: [▼ all]  Modality: [▼ all]              │   │
│  │  Organ: [___________]  Search: [___________]      │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ kidney-seg-task              CT · Abdomen_Kidney │   │
│  │ macro-Dice  │  S1 S2 S3 │ 7 agents · 140 runs    │   │
│  │ [=====A====] │ [● ● ●  ] │  Lite ✓  Std ✓  Pro —│   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ liver-seg-task              CT · Abdomen_Liver   │   │
│  │ macro-Dice  │  S1 S2 S3 │ 7 agents · 140 runs    │   │
│  │ [====B=====] │ [● ● ●  ] │  Lite ✓  Std ✓  Pro —│   │
│  └──────────────────────────────────────────────────┘   │
│  ... (56 more task cards)                                │
└──────────────────────────────────────────────────────────┘
```

**Core functionality:**
- Filterable card grid (domain, modality, organ, search)
- Each card shows:
  - Task name + organ + modality
  - Metric type badge
  - Skill coverage dots: S1 ● S2 ● S3 (filled = skill file exists, empty = none)
  - Tier status: Lite ✓ / Std ✓ / Pro — (check = has runs)
  - Agent count + total runs
  - Best overall score bar
  - Top agent name
- Click card → expands to show config.yaml + model_info.yaml preview
- "Open in Skill Browser" link

---

## Part C: Navigation Structure

### Public Site Nav
```
┌────────────────────────────────────────────┐
│ [AutoMedBench]  Leaderboard  Tasks  Docs   │
└────────────────────────────────────────────┘
```

### Ops Console Nav
```
┌──────────────────────────────────────────────────────────┐
│ [AutoMedBench OPS]  Matrix  Tasks  Skills  Rubrics       │
│                     Harness  Runs                        │
└──────────────────────────────────────────────────────────┘
```

The Ops Console doesn't need to be public. It can be a separate local dev server or a password-protected section of the site.

---

## Part D: Cross-Linking Map

Every UI connects to related UIs:

```
Score Matrix ──click cell──▶ Run Inspector
     │                            │
     ├──click agent──▶ Agent Profile
     │                        │
     └──click task───▶ Task Detail ──▶ Skill Browser
                              │              │
                              └──▶ Rubric Explorer
                                        │
                                        └──▶ Harness Viewer
```

---

## Part E: Technical Implementation Notes

### Data Pipeline
```
Task dirs ──▶ Build script ──▶ Static JSON ──▶ Site/Console
  (58+)      (python)          (per domain)    (Next.js SSG)
```

A build script reads all `detail_report.json` files + `config.yaml` + skill files and produces:
- `public/data/leaderboard.json` — aggregated rankings
- `public/data/tasks.json` — task index with metadata
- `public/data/agents.json` — agent profiles
- `public/data/matrix.json` — score matrix data
- `public/data/skills.json` — skill file contents indexed by task+tier+stage

### Tech Stack (same for both sites)
- Next.js App Router (static export for public site, optional server for ops console)
- Tailwind CSS + CSS custom properties for theming
- Framer Motion for page transitions and pipeline animation
- D3.js lazy-loaded for radar, scatter, histogram
- Code highlighting: Shiki (for skill files and code blocks)

### Static Export Strategy
Public site → fully static, deploy to GitHub Pages.
Ops Console → static too, but data files are larger and need a build step. Could also run as a local dev server pointed at the workspace data directory.

---

## Part F: Implementation Priority

### Phase 1 — Public MVP
- Landing page
- Leaderboard (global)
- Task detail with S1-S5 pipeline
- Basic responsive

### Phase 2 — Ops Core
- Score Matrix (biggest value — see everything at once)
- Run Inspector (debugging)
- Task Manager

### Phase 3 — Depth
- Skill Browser
- Rubric Explorer
- Harness Viewer
- Agent Profile + Compare (public)
- Domain leaderboards

### Phase 4 — Polish
- Documentation pages
- Search
- Accessibility audit
- Light mode
