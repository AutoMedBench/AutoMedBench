# Detailed Rubrics

Detailed judge-facing rubric for `eval_report_gen`.

## Scoring Convention

- The judge resolves absolute artifact paths from the run record.
- Every relative path below is relative to `OUTPUT_DIR = <run_dir>/outputs`.
- `plan/plan.md` means `OUTPUT_DIR/plan/plan.md`.
- `plan/plan.png` means `OUTPUT_DIR/plan/plan.png`.
- `agent_outputs/<case_id>/report.txt` means `OUTPUT_DIR/agent_outputs/<case_id>/report.txt`.
- `S1` through `S3` are evaluated as binary atomic checks.
- Score each question as `1` if the artifact or execution trace clearly satisfies it, else `0`.
- `S1`, `S2`, and `S3` are the mean of their applicable binary questions.
- Tier-specific or conditional questions are excluded from the denominator when not applicable.
- A downstream judge may still bucket the final step score into coarse bands, but the review itself should happen at the question level.

## Benchmark Terms

- `case`: the benchmark input unit, stored under `public/<case_id>/`. In the source dataset this corresponds to one study, but this rubric uses `case` throughout.
- `case images`: the one or more JPEG files listed in `public/<case_id>/manifest.json["image_files"]`.
- `selected inference entrypoint`: the exact repo-defined script, notebook, CLI, or `python -m ...` path chosen in `plan.md` for the selected method.
- `official export logic`: the selected method's own documented decode / post-process / section-extraction path that turns raw model output into the final report text. Examples include repo helpers such as `split_and_decode_sections(...)`, `decode_sections(...)`, a documented `section_parser`, or notebook cells that perform the official post-processing. If the selected method has no documented export helper, `plan.md` should say so explicitly and describe the minimal adapter.

Source anchors for the case-unit checks:

- `eval_report_gen/README.md`: "One benchmark case is one study, not one individual JPEG."
- `eval_report_gen/mimic-cxr-report-task/config.yaml`: "Generate one findings-only chest X-ray report from one or more JPEG views."
- `eval_report_gen/mimic-cxr-report-task/README.md`: a case may contain one or more staged JPEG files.

## Workflow / Agentic Score (50%)

### S1 Plan (25% of workflow)

Evidence sources:

- `OUTPUT_DIR/plan/plan.md`
- `OUTPUT_DIR/plan/plan.png` for Standard and Pro

Binary questions:

- `S1.Q1`: Is `OUTPUT_DIR/plan/plan.md` present?
- `S1.Q2`: Does `plan.md` name the selected model?
- `S1.Q3`: Does `plan.md` name the selected inference entrypoint?
- `S1.Q4`: Does `plan.md` reflect the benchmark unit correctly: one case is one full case input, not one JPEG?
- `S1.Q5`: Does `plan.md` state that all files in `public/<case_id>/manifest.json["image_files"]` must be used?
- `S1.Q6`: Does `plan.md` state the exact output path `OUTPUT_DIR/agent_outputs/<case_id>/report.txt`?
- `S1.Q7`: Does `plan.md` require pilot validation before full-batch generation?
- `S1.Q8`: Does `plan.md` define at least one specific pilot check?
- `S1.Q9`: Standard/Pro only: is `OUTPUT_DIR/plan/plan.png` present?
- `S1.Q10`: Standard/Pro only: does `plan.png` show the same selected model and the same stage order as `plan.md`?
- `S1.Q11`: Standard/Pro only: does `plan.md` compare multiple candidate methods and justify the selected method?

Aggregation:

- `S1 = mean(applicable S1.Q*)`

### S2 Setup (15% of workflow)

Evidence sources:

- execution trace
- created code and scripts
- saved pilot outputs

Binary questions:

- `S2.Q1`: Does successful execution run the selected inference pipeline?
- `S2.Q2`: Does setup read `public/<case_id>/manifest.json`?
- `S2.Q3`: Does setup load case images from `manifest.json["image_files"]`?
- `S2.Q4`: Does setup preserve one case as one input unit by grouping all JPEGs listed in `manifest.json["image_files"]` for a single report, rather than treating each JPEG as its own case?
- `S2.Q5`: Does setup write `OUTPUT_DIR/agent_outputs/<case_id>/report.txt`?
- `S2.Q6`: Does setup use the exact selected method entrypoint named in `S1`?
  Ground-truth expectation: the answer should name the concrete entrypoint chosen in `plan.md`, for example an official script like `python3 main_v0926_ablation_study.py`, an official `bash run_cxr_*.sh`, a documented notebook, or a documented `python -m ...` command.
- `S2.Q7`: If the selected method defines official export logic, does setup use that exact decode / post-process path when producing `report.txt`?
  Ground-truth expectation: the answer should point to the concrete repo-defined helper or notebook path. If the method has no documented export helper, this question is conditional and `plan.md` should explicitly note that no official export path exists before a minimal adapter is introduced.

Aggregation:

- `S2 = mean(applicable S2.Q*)`

### S3 Validate (35% of workflow)

Evidence sources:

- execution trace
- saved pilot outputs
- validation commands or inspection steps

Binary questions:

- `S3.Q1`: Does at least one pilot case run before full-batch generation?
- `S3.Q2`: Does the pilot run use the same selected model, checkpoint, inference entrypoint, and export/decode path as `S2`, rather than a separate toy or validation-only path?
- `S3.Q3`: Does the pilot run write `OUTPUT_DIR/agent_outputs/<case_id>/report.txt`?
- `S3.Q4`: After the pilot run, is the saved `report.txt` opened, read, or otherwise directly inspected?
- `S3.Q5`: Does validation check `manifest.json["image_files"]`?
- `S3.Q6`: Does validation confirm that all JPEGs listed in `manifest.json["image_files"]` for that case were actually used?
- `S3.Q7`: If the first pilot run fails, does a later pilot rerun occur before full-batch generation?

Aggregation:

- `S3 = mean(applicable S3.Q*)`

### S4 Inference (15% of workflow)

- `s4 = 0.50 * completion_rate + 0.50 * format_valid`
- `completion_rate = success_count / sample_count`
- `format_valid = 1` if all outputs are format-valid, else `0`
- Example: `9/10` complete and format-valid -> `0.50 * 0.9 + 0.50 * 1 = 0.95`

### S5 Submit (10% of workflow)

- `s5 = 0.50 * any_valid_results + 0.50 * format_valid`
- `any_valid_results = 1` if at least one valid result exists, else `0`
- `format_valid = 1` if the batch output format is valid, else `0`
- Example: some valid results but invalid batch format -> `0.50`

### Workflow Aggregate

- `workflow = 0.25*S1 + 0.15*S2 + 0.35*S3 + 0.15*S4 + 0.10*S5`
- Active steps are normalized in `aggregate.py` when the workflow score is built.

## Clinical Score (50%)

- Default task config uses 7 equally weighted MLRG components:
- `BLEU`
- `METEOR`
- `ROUGE_L`
- `F1RadGraph`
- `micro_average_precision`
- `micro_average_recall`
- `micro_average_f1`
- If the batch is incomplete or format-invalid, the clinical score is forced to `0`.

## Final Aggregation

- `overall = 0.50 * workflow + 0.50 * clinical`
- Output only the continuous `overall` score.

Implementation anchors:

- `eval_report_gen/aggregate.py`
- `eval_report_gen/llm_judge.py`
- `eval_report_gen/README.md`
