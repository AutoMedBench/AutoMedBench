# eval_report_gen

Study-level chest X-ray report-generation benchmark package.

This package is the report-generation sibling of `eval_seg/`. Its default task
is `mimic-cxr-report-task`, backed by a 10-study pilot split staged from:

- `/mnt/data-2u-2/jmao/mimic-cxr/test`

## Benchmark Unit

One benchmark case is one study, not one individual JPEG.

Each study may contain one or more views:

- `public/<case_id>/images/*.jpg`
- `public/<case_id>/manifest.json`

The private reference side contains:

- `private/<case_id>/report.txt`
- `private/<case_id>/labels.json`
- `private/<case_id>/manifest.json`

## Scoring

The report-side score is configurable. The default task config now uses the
`MLRG` metric stack and aggregates:

- `BLEU` (mean of BLEU-1 through BLEU-4)
- `METEOR`
- `ROUGE_L`
- `F1RadGraph`
- `micro_average_precision`
- `micro_average_recall`
- `micro_average_f1`

All seven components are equally weighted by default. The legacy lightweight
fallback scorer is still available for environments without the heavy metric
dependencies.

Overall score is:

- `0.5 * workflow`
- `0.5 * clinical`

Ratings:

- `A`: good result
- `B`: okay result
- `C`: below baseline
- `F`: invalid or incomplete output

## Layout

- `benchmark_runner.py`: execute-code benchmark harness with baseline support
- `run_eval.py`: evaluation entry point
- `report_scorer.py`: deterministic label extraction and scoring
- `format_checker.py`: `report.txt` validation
- `aggregate.py`: workflow + clinical aggregation
- `llm_judge.py`: heuristic/online S1-S3 judge
- `stage_data.py`: pilot staging from flat MIMIC-CXR files
- `task_loader.py`: config and task discovery
- `tier_config.py`: Lite / Standard / Pro presets
- `mimic-cxr-report-task/`: task-specific config, model info, and skill blocks
- `baselines/`: `empty`, `constant_normal`, and `perfect` reference outputs
- `results/`: sample benchmark run outputs and copied result JSONs
- `tests/`: local unit and smoke tests
- `docker/`: container orchestration and mount-separation tests

## Local Credentials

The local benchmark-agent credential source is:

- `eval_report_gen/api_keys/key.txt`

That file must contain exactly four lines, with one API key on each line and
no labels, comments, or example code. NVIDIA-backed agents read and rotate
through those four keys from the file instead of scraping free-form text.

Example format:

```text
sk-...
sk-...
sk-...
sk-...
```

The active benchmark-agent config is duplicated locally in:

- `eval_report_gen/agent_config.yaml`
