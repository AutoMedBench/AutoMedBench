# Docs

High-level architectural notes for `eval_report_gen`.

## Task

- study-level chest X-ray report generation
- one or more JPEG views per case
- one `report.txt` output per case

## Workflow Rubric

- `S1 Plan`: research/reporting strategy and artifact creation
- `S2 Setup`: establish a working generation pipeline
- `S3 Validate`: validate on pilot studies before batch generation
- `S4 Inference`: generate reports for every case
- `S5 Submit`: verify outputs and submit the batch

## Clinical Rubric

- fixed chest X-ray observation ontology
- deterministic extraction from reference and predicted reports
- observation F1 + ROUGE-L report similarity

## Default Judge Behavior

- offline/default: heuristic judge
- optional online mode: OpenAI-compatible judge call through local key file

## Agent Config Decision

`eval_report_gen` keeps its own local `agent_config.yaml` instead of sharing
`eval_seg/agent_config.yaml`.

Reason:

- report generation has a separate credential path
- it needs local baseline agents
- it should evolve independently from segmentation-agent config

## Docker Safety Invariant

- agent container mounts only staged `public/`
- eval container mounts `public/`, `private/`, and read-only agent outputs
