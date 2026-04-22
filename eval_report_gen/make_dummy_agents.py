#!/usr/bin/env python3
"""Build dummy baselines for eval_report_gen."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from task_loader import discover_cases, get_task_data_root, load_task_config


CONSTANT_NORMAL_REPORT = """FINAL REPORT
FINDINGS: Portable chest radiograph demonstrates no focal airspace consolidation. No pleural effusion or pneumothorax. Cardiomediastinal silhouette is stable.
IMPRESSION: No acute cardiopulmonary process.
"""


def _baseline_root(out_dir: str | Path) -> Path:
    root = Path(out_dir)
    (root / "agent_outputs").mkdir(parents=True, exist_ok=True)
    return root


def build_empty(task_id: str, out_dir: str | Path) -> Path:
    root = _baseline_root(out_dir)
    for case_id in discover_cases(task_id):
        case_dir = root / "agent_outputs" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "report.txt").write_text("", encoding="utf-8")
    return root


def build_constant_normal(task_id: str, out_dir: str | Path) -> Path:
    root = _baseline_root(out_dir)
    for case_id in discover_cases(task_id):
        case_dir = root / "agent_outputs" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "report.txt").write_text(CONSTANT_NORMAL_REPORT, encoding="utf-8")
    return root


def build_perfect(task_id: str, out_dir: str | Path) -> Path:
    root = _baseline_root(out_dir)
    data_root = Path(get_task_data_root(task_id))
    private_root = data_root / "private"
    for case_id in discover_cases(task_id):
        case_dir = root / "agent_outputs" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(private_root / case_id / "report.txt", case_dir / "report.txt")
    return root


def build_baseline(task_id: str, baseline: str, out_dir: str | Path) -> Path:
    if baseline == "empty":
        return build_empty(task_id, out_dir)
    if baseline == "constant_normal":
        return build_constant_normal(task_id, out_dir)
    if baseline == "perfect":
        return build_perfect(task_id, out_dir)
    raise ValueError(f"Unknown baseline '{baseline}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dummy baselines for eval_report_gen")
    parser.add_argument("--task", default="mimic-cxr-report-task")
    parser.add_argument("--baseline", required=True, choices=["empty", "constant_normal", "perfect"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = build_baseline(args.task, args.baseline, args.out_dir)
    print(f"Built {args.baseline} baseline -> {out}")


if __name__ == "__main__":
    main()
