# How to Prompt Experiments

Copy-paste prompts for launching benchmark runs in Claude Code.
Two roles: **Experimenter** (runs agents) and **Debugger** (finds bugs).

---

## Experimenter Prompt

Use this to benchmark one or more agents. Fill in the 6 fields, paste into Claude Code.

```
cd ./eval_seg

Read bench-kidney-standard.skill and benchmark.skill in this project.

My experimenter name: <NAME>
Agent to benchmark:   <AGENT_KEY>
Task:                 <kidney|liver|pancreas>
Tier:                 <lite|standard|pro>
Judge:                Online Opus 4.6
Where to save:        ./eval_seg/runs/<NAME>/bench-<AGENT_SHORT>-<TASK>-<TIER>

Verify the API works, then run the benchmark. Pass --output-dir with the
"Where to save" path so all outputs land there directly. Monitor only your
own run until it finishes and report results. If you hit an API rate limit,
stop and tell me. Do NOT kill any other processes — they may be parallel
experiments.

DO NOT TRY TO CARE ABOUT OTHERS. e.g. kill other process or monitor other
process info.
Launch <N> tests one-by-one. Monitor only, tell me when all finish. Make
the submission in one bash so I do not need to confirm "enter" in next
task submission after one finished.

Display progress in this form:

<N> tasks (1 done, 1 in progress, <N-2> open)
  ✔ Run 1 of <N>: <agent> <task>…
  ◼ Run 2 of <N>: <agent> <task>…
  ◻ Run 3 of <N>: <agent> <task>…

Only display and tell me when all finish.
```

### Example: 2 runs of Gemini 3.1 Pro on kidney standard

```
cd ./eval_seg

Read bench-kidney-standard.skill and benchmark.skill in this project.

My experimenter name: kuma
Agent to benchmark:   gemini-3.1-pro
Task:                 kidney
Tier:                 standard
Judge:                Online Opus 4.6
Where to save:        ./eval_seg/runs/kuma/bench-gemini3.1pro-kidney-standard

Verify the API works, then run the benchmark. Pass --output-dir with the
"Where to save" path so all outputs land there directly. Monitor only your
own run until it finishes and report results. If you hit an API rate limit,
stop and tell me. Do NOT kill any other processes — they may be parallel
experiments.

DO NOT TRY TO CARE ABOUT OTHERS. e.g. kill other process or monitor other
process info.
Launch 2 tests one-by-one. Monitor only, tell me when all finish. Make
the submission in one bash so I do not need to confirm "enter" in next
task submission after one finished.

Display progress in this form:

2 tasks (1 done, 1 in progress, 0 open)
  ✔ Run 1 of 2: gemini-3.1-pro kidney…
  ◼ Run 2 of 2: gemini-3.1-pro kidney…

Only display and tell me when all finish.
```

---

## Debugger Prompt

Use this to replay failing runs, find root causes, and file bug tickets.

```
cd ./eval_seg

Read bench-kidney-standard.skill and bug_issues/ in this project.

My experimenter name: debugger
Agent to benchmark:   <AGENT_KEY>
Task:                 <kidney|liver|pancreas>
Tier:                 <lite|standard|pro>
Judge:                Online Opus 4.6
Where to save:        ./eval_seg/runs/debugger/bench-<AGENT_SHORT>-<TASK>-<TIER>
Status tracking:      Yes, refer to benchmark.skill
File Tickets:         Yes, refer to bug_issues/
Debug Type:           Naive errors that stuck the agents (priority); other bugs as well.

Verify the API works, then run the benchmark. Pass --output-dir with the
"Where to save" path so all outputs land there directly. Monitor only your
own run until it finishes and report results. If you hit an API rate limit,
stop and tell me. Do NOT kill any other processes — they may be parallel
experiments.

DO NOT TRY TO CARE ABOUT OTHERS. e.g. kill other process or monitor other
process info.
```

### What the debugger does

1. Runs the agent the same way an experimenter would
2. Reads `bug_issues/` to understand known bugs and ticket format
3. When the agent fails, investigates root cause
4. Files a ticket in `bug_issues/` following the existing format:

```
bug_issues/<NNN>_<short_description>.md
```

Each ticket has frontmatter (`id`, `title`, `severity`, `affected`, `discovered`, `run`) plus Summary, Root Cause, and Fix sections. Look at existing tickets for the format.

### Example: debugging kidney standard

```
cd ./eval_seg

Read bench-kidney-standard.skill and bug_issues/ in this project.

My experimenter name: debugger
Agent to benchmark:   gemini-3.1-pro
Task:                 kidney
Tier:                 standard
Judge:                Online Opus 4.6
Where to save:        ./eval_seg/runs/debugger/bench-gemini3.1pro-kidney-standard
Status tracking:      Yes, refer to benchmark.skill
File Tickets:         Yes, refer to bug_issues/
Debug Type:           Naive errors that stuck the agents (priority); other bugs as well.

Verify the API works, then run the benchmark. Pass --output-dir with the
"Where to save" path so all outputs land there directly. Monitor only your
own run until it finishes and report results. If you hit an API rate limit,
stop and tell me. Do NOT kill any other processes — they may be parallel
experiments.

DO NOT TRY TO CARE ABOUT OTHERS. e.g. kill other process or monitor other
process info.
```

---

## Fields reference

| Field | Where to find valid values |
|-------|---------------------------|
| Agent key | `eval_seg/agent_config.yaml` — use the YAML key (e.g., `gemini-3.1-pro`, `claude-opus-4-6`, `qwen3.5-397b`) |
| Task | `kidney`, `liver`, `pancreas` — or any `<task-id>` folder under `eval_seg/` |
| Tier | `lite`, `standard`, `pro` |
| Judge | Online Opus 4.6 (default) or `--offline-judge` for local DeepSeek |
| Where to save | Convention: `./eval_seg/runs/<name>/bench-<agent_short>-<task>-<tier>` |

---

## Tips

- **Multiple agents in one session**: change the agent key and "Where to save" path between runs. Each gets its own directory.
- **Multiple tasks**: change task + save path. E.g., run kidney then liver for the same agent.
- **The runner auto-appends a `YYMMDD-6hex` run tag** inside the save directory. Runs never collide even if you launch the same combo twice.
- **tracker.md**: the benchmark.skill tells the agent to maintain a tracker file in the save directory. Check it for run history.
- **Rate limits (429)**: the prompt tells the agent to stop and report. Do NOT let it retry in a loop.
