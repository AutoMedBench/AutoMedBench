#!/usr/bin/env bash
set -euo pipefail

cd /app
python eval_report_gen/benchmark_runner.py \
  --agent "${AGENT}" \
  --task "${TASK}" \
  --tier "${TIER}" \
  --offline-judge \
  --output-dir "${OUTPUT_DIR}"
