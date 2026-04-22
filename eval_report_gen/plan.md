# eval_report_gen Plan

This file is the execution anchor for the report-generation benchmark.

Rule: check an item only when the corresponding files or behavior exist in the
repo and have been verified.

## 0. Foundation

- [x] Create the sibling benchmark package `eval_report_gen/`
- [x] Create the task folder `mimic-cxr-report-task/`
- [x] Seed a 10-study pilot split in `splits/pilot_10_studies.txt`
- [x] Add placeholder benchmark module entry points
- [x] Add top-level `docs/`, `tests/`, `baselines/`, `results/`, and `docker/` scaffolding

## 1. Study Staging

- [x] Define the canonical study ID format and raw-file grouping rule
- [x] Define the staged `public/` and `private/` layout for one study
- [x] Implement `stage_data.py` for the 10-study pilot
- [x] Emit a pilot-level `manifest.csv`
- [x] Emit a per-study `manifest.json`
- [x] Verify that each pilot case maps to one study with one or more images and exactly one target report

## 2. Task Contract

- [x] Decide the required agent inputs under `public/`
- [x] Decide the required outputs under `agent_outputs/`
- [x] Finalize `mimic-cxr-report-task/config.yaml`
- [x] Finalize the v1 output contract for `report.txt`
- [x] Define completion, missing-output, and invalid-output rules

## 3. Clinical Rubric

- [x] Choose the v1 deterministic clinical metrics
- [x] Define the observation ontology or label schema
- [x] Decide how ground-truth structured labels are derived from the reference reports
- [x] Decide how predicted structured labels are derived from generated reports
- [x] Set pilot thresholds for `A`, `B`, `C`, and `F`
- [x] Rewrite the failure taxonomy for report-generation errors

## 4. Workflow Rubric

- [x] Rewrite the `S1` to `S5` semantics for report generation
- [x] Define `S3` pilot-validation expectations
- [x] Define `S4` batch-generation expectations
- [x] Define `S5` submission expectations
- [x] Define the report-generation detail report sections

## 5. Levels

- [x] Lite: define exact model guidance and provided requirements
- [x] Lite: define the `S1` to `S3` skill content
- [x] Standard: define model-family guidance and `S1` / `S3` skill content
- [x] Pro: define open-research prompt behavior and artifact expectations
- [x] Confirm whether `plan.png` is required per tier
- [x] Confirm whether summary plots are required in Pro

## 6. Evaluation Pipeline

- [x] Implement `format_checker.py`
- [x] Add a report-scoring module
- [x] Implement `aggregate.py`
- [x] Implement `failure_classifier.py`
- [x] Implement `detail_report.py`
- [x] Implement `run_eval.py`
- [x] Save one or more sample result JSON files in `results/`

## 7. Runner and Prompting

- [x] Adapt `benchmark_runner.py` to report-generation task flow
- [x] Implement `task_loader.py` for report tasks
- [x] Decide whether `agent_config.yaml` is duplicated or shared
- [x] Finalize the task prompt files under `mimic-cxr-report-task/`
- [x] Verify pilot run-directory structure and artifact archiving

## 8. LLM Judge

- [x] Rewrite `llm_judge.py` prompts and rubrics for report generation
- [x] Define the judge inputs from conversation plus eval report
- [x] Add parser or contract tests for judge output
- [x] Decide the default online and offline judge behavior

## 9. Baselines and Tests

- [x] Add an `empty` baseline
- [x] Add a `constant_normal` baseline
- [x] Add a `perfect` baseline
- [x] Add unit tests for study grouping and output-format validation
- [x] Add an evaluator smoke test on the 10-study pilot split
- [x] Add an LLM-judge smoke test

## 10. Docker and Isolation

- [x] Port the host runner to Docker agent and eval images
- [x] Revisit isolation rules for JPEG and report tasks
- [x] Add container integration tests
- [x] Confirm that private reports cannot leak into agent-visible paths

## Working Rules

- [x] Always treat this file as the source of truth for `eval_report_gen`
- [x] Refine one stage or one tier level at a time unless the user asks otherwise
- [x] Keep default scope to the 10-study pilot unless explicitly expanded
- [x] Update this checklist in the same turn when work is completed
