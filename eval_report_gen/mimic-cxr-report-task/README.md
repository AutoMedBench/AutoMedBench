# mimic-cxr-report-task

Task-specific configuration for study-level chest X-ray report generation.

This task folder is where we will refine:

- public/private case layout
- tier-specific instructions
- model guidance
- task-specific requirements
- report-generation rubric assumptions

## Raw Grouping Rule

The raw source directory is a flat file tree. Each file name follows:

`pXX_pXXXXXXXX_sXXXXXXXX_<uuid>.<ext>`

The canonical raw study ID is the first three underscore-delimited fields:

`pXX_pXXXXXXXX_sXXXXXXXX`

All JPEG and TXT files sharing that prefix belong to one study-level case.

## Pilot Staged Layout

The pilot stager materializes benchmark-ready data under:

`data/MIMIC_CXR_Report/pilot_10/`

```text
data/MIMIC_CXR_Report/pilot_10/
  public/
    CXR0001/
      images/
        01.jpg
        02.jpg
      manifest.json
  private/
    CXR0001/
      report.txt
      manifest.json
  manifest.csv
```

### Public Side

- `CXR0001`, `CXR0002`, ... are neutral benchmark case IDs
- `images/` contains one or more staged JPEG views for the study
- `manifest.json` contains agent-safe metadata such as `case_id` and image count

### Private Side

- `report.txt` is the single canonical findings-only reference text for the study
- `labels.json` stores deterministic structured observation labels derived from `report.txt`
- `manifest.json` records the raw study ID, source file names, and report hash

### Pilot Guarantee

For each staged pilot case:

- one raw study ID maps to one benchmark case
- the case has one or more JPEG views
- all paired raw TXT files are identical
- the staged reference text is extracted from the raw `FINDINGS` section only
- studies without a non-empty `FINDINGS` section are not valid for this split
- exactly one canonical findings-only reference report is staged into `private/`

## Task Contract

### Agent Inputs

Agents should read only the staged `public/` side:

- `public/<case_id>/images/*.jpg`
- `public/<case_id>/manifest.json`

The manifest is agent-safe and may be used to determine the study ID and image count.

### Required Outputs

Agents must write:

- `agent_outputs/<case_id>/report.txt`

Exactly one `report.txt` is required per case. The benchmark scores only the
`FINDINGS` content:

- if the model writes a `FINDINGS:` section, only that section is scored
- if the model writes plain text with no section headers, the whole file is treated as findings text
- `IMPRESSION` is ignored by the scorer

### Valid `report.txt`

A valid report must:

- be UTF-8 decodable text
- contain at least 40 characters
- contain at least 20 alphabetic characters
- stay within the configured max length
- be non-empty after stripping whitespace

### Completion Rules

- every staged case must have a valid `report.txt`
- any missing or invalid case forces the run to `F`
- partial completion is tracked diagnostically, but clinical score is zeroed

## Clinical Rubric

The default task config uses the `MLRG` metric stack for the report-side score.
It aggregates:

- `BLEU` (defined as the mean of BLEU-1 through BLEU-4)
- `METEOR`
- `ROUGE_L`
- `F1RadGraph`
- `micro_average_precision`
- `micro_average_recall`
- `micro_average_f1`

All components are equally weighted by default. This backend requires the
external `MLRG` repo plus local CheXbert, BERT, and RadGraph checkpoints.

The original lightweight scorer is still implemented as a fallback backend for
debugging and low-dependency runs.

## Workflow Rubric

### S1 Plan

- research the reporting pipeline or model family
- write `plan/plan.md`
- in Standard and Pro, also provide `plan/plan.png`

### S2 Setup

- establish a working code path to load study images and generate reports
- confirm the output contract can be written correctly

### S3 Validate

- run one or more pilot studies first
- inspect saved `report.txt`
- confirm that all study views were used

### S4 Inference

- generate valid reports for every staged case
- partial completion is tracked diagnostically but fails the run

### S5 Submit

- verify that every case has a valid `report.txt`
- then submit the batch

## Tier Semantics

### Lite

- guided model choice
- provided `requirements.txt`
- skill guidance for `S1` through `S3`
- `plan.md` required
- `plan.png` not required

### Standard

- compare multiple model families
- `plan.md` and `plan.png` required
- skill guidance for `S1` and `S3`

### Pro

- open research with minimal guidance
- `plan.md` and `plan.png` required
- summary plots enabled in the final detail report package
- the agent is expected to justify why its chosen reporting pipeline is better
  than at least two alternatives

## How To Run

Run the benchmark from the repository root:

```bash
cd /mnt/data-2u-2/jmao/MedAgentsBench-report-gen
python3 eval_report_gen/benchmark_runner.py \
  --agent claude-opus-4-6 \
  --task mimic-cxr-report-task \
  --tier lite
```

Run the Standard tier from the repository root:

```bash
cd /mnt/data-2u-2/jmao/MedAgentsBench-report-gen
python3 eval_report_gen/benchmark_runner.py \
  --agent claude-opus-4-6 \
  --task mimic-cxr-report-task \
  --tier standard
```

To place outputs under a fixed directory:

```bash
cd /mnt/data-2u-2/jmao/MedAgentsBench-report-gen
python3 eval_report_gen/benchmark_runner.py \
  --agent claude-opus-4-6 \
  --task mimic-cxr-report-task \
  --tier lite \
  --output-dir /mnt/data-2u-2/jmao/MedAgentsBench-report-gen/eval_report_gen/runs/claude-opus4.6-pilot10-lite-continue
```

To place Standard outputs under a fixed directory:

```bash
cd /mnt/data-2u-2/jmao/MedAgentsBench-report-gen
python3 eval_report_gen/benchmark_runner.py \
  --agent claude-opus-4-6 \
  --task mimic-cxr-report-task \
  --tier standard \
  --output-dir /mnt/data-2u-2/jmao/MedAgentsBench-report-gen/eval_report_gen/runs/claude-opus4.6-pilot10-standard
```

To use the local heuristic judge instead of the online judge:

```bash
cd /mnt/data-2u-2/jmao/MedAgentsBench-report-gen
python3 eval_report_gen/benchmark_runner.py \
  --agent claude-opus-4-6 \
  --task mimic-cxr-report-task \
  --tier lite \
  --offline-judge
```

To use the local heuristic judge for Standard:

```bash
cd /mnt/data-2u-2/jmao/MedAgentsBench-report-gen
python3 eval_report_gen/benchmark_runner.py \
  --agent claude-opus-4-6 \
  --task mimic-cxr-report-task \
  --tier standard \
  --offline-judge
```

### Preconditions

- staged pilot data must exist under:
  `data/MIMIC_CXR_Report/pilot_10/`
- API keys must be available in:
  `eval_report_gen/api_keys/key.txt`
  with exactly four lines and one key per line
- agent definitions live in:
  `eval_report_gen/agent_config.yaml`

### Available Agents

- `claude-opus-4-6`
- `claude-sonnet-4-6`
- `baseline-empty`
- `baseline-constant-normal`
- `baseline-perfect`

### Main Outputs

After a run finishes, inspect:

- `eval_report_gen/runs/.../detail_report.json`
- `eval_report_gen/runs/.../eval_report.json`
- `eval_report_gen/runs/.../process/trace.jsonl`
- `eval_report_gen/runs/.../process/tool_calls.jsonl`
- `eval_report_gen/runs/.../process/conversation.json`

Generated case reports are written to:

- `eval_report_gen/runs/.../outputs/agent_outputs/<case_id>/report.txt`

Current validated Standard run:

- `eval_report_gen/runs/claude-opus4.6-pilot10-standard/260415-dde060`
