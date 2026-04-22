#!/usr/bin/env python3
"""Back-fill agentic_score / step_scores / overall_score into committed
detail_report.json files.

Ports eval_seg/aggregate.py formulas exactly (commit-for-commit weights and
S1-S3 = None stubs), so agentic numbers are comparable across sibling
domains. No re-run required — all inputs come from the existing report's
`format`, `scores`, `pass_rate`, and `clinical` sections.

Formulas — canonical rubric in `../RUBRIC.md` (harness-scored rows only):
    completion_rate   = n_valid / n_patients              # S4 completeness
    format_valid      = format.output_format_valid         # S4 / S5 format gate
    has_valid_results = (n_valid > 0) and (mean_psnr is finite)  # S5 content gate
    s1 = s2 = s3 = None                                   # reserved for LLM judge
    s4 = 0.50*completion_rate  + 0.50*format_valid
    s5 = 0.50*has_valid_results + 0.50*format_valid
    agentic_score  = Σ w[k] * step_scores[k] / Σ w   (S1-S3=None skipped)
        where w = {s1:0.25, s2:0.15, s3:0.35, s4:0.15, s5:0.10}
    clinical_score = clinical.clinical (already per-task normalized)
    overall_score  = 0.50 * agentic + 0.50 * clinical

All three scores are written into each detail_report.json under a new
`aggregate` block, leaving existing fields untouched so aggregate_v5.py
and the existing figures continue to work.

Usage:
    python compute_agentic_score.py [run_root=results/matrix_v5_full]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STEP_WEIGHTS = {"s1": 0.25, "s2": 0.15, "s3": 0.35, "s4": 0.15, "s5": 0.10}


def compute_workflow_score(step_scores: dict) -> tuple[float, list[str]]:
    """Weighted average; None steps are skipped and their weight redistributes."""
    active = {k: v for k, v in step_scores.items() if v is not None}
    if not active:
        return 0.0, []
    active_w = {k: STEP_WEIGHTS[k] for k in active}
    total_w = sum(active_w.values())
    score = sum(active_w[k] * active[k] for k in active) / total_w
    return round(score, 4), sorted(active.keys())


def back_fill_one(report_path: Path) -> bool:
    try:
        r = json.load(open(report_path))
    except Exception:
        return False

    fmt = r.get("format") or {}
    scr = r.get("scores") or {}
    clin = r.get("clinical") or {}
    orch = r.get("orchestrator") or {}

    n_valid = fmt.get("n_valid", 0) or 0
    n_patients = fmt.get("n_patients") or r.get("n_patients") or 0
    output_format_valid = bool(fmt.get("output_format_valid", False))

    # Isolation breach zeroes everything (match seg orchestrator penalty)
    if orch.get("isolation_breach"):
        step_scores = {"s1": None, "s2": None, "s3": None,
                       "s4": 0.0, "s5": 0.0}
        agentic_score = 0.0
        clinical_score = 0.0
        overall_score = 0.0
    elif n_patients == 0:
        # Skeleton / unscorable — leave as F-equivalent
        step_scores = {"s1": None, "s2": None, "s3": None,
                       "s4": 0.0, "s5": 0.0}
        agentic_score = 0.0
        clinical_score = 0.0
        overall_score = 0.0
    else:
        import math
        completion_rate = n_valid / n_patients
        mean_psnr = scr.get("mean_psnr") if isinstance(scr, dict) else None
        has_valid_results = (n_valid > 0) and (mean_psnr is not None) \
                            and isinstance(mean_psnr, (int, float)) \
                            and math.isfinite(float(mean_psnr))
        s4 = round(0.50 * completion_rate + 0.50 * float(output_format_valid), 4)
        s5 = round(0.50 * float(has_valid_results) + 0.50 * float(output_format_valid), 4)
        step_scores = {"s1": None, "s2": None, "s3": None, "s4": s4, "s5": s5}
        agentic_score, _ = compute_workflow_score(step_scores)
        clinical_score = float(clin.get("clinical", 0.0))
        overall_score = round(0.50 * agentic_score + 0.50 * clinical_score, 4)

    r["step_scores"] = step_scores
    r["aggregate"] = {
        "agentic_score":  agentic_score,
        "clinical_score": clinical_score,
        "overall_score":  overall_score,
        "rating":         (r.get("rating") or {}).get("rating", "F"),
    }

    with open(report_path, "w") as f:
        json.dump(r, f, indent=2, default=str)
    return True


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else str(Path(__file__).resolve().parent.parent
                         / "results" / "matrix_v5_full")).resolve()
    # Fallback to runs/ tree if given runs path
    if not root.is_dir():
        alt = (Path(__file__).resolve().parent.parent.parent
               / "runs" / root.name)
        if alt.is_dir():
            root = alt
    if not root.is_dir():
        print(f"ERROR: {root} not found"); return 2

    updated = 0
    failed = 0
    for p in root.rglob("detail_report.json"):
        if back_fill_one(p):
            updated += 1
        else:
            failed += 1
    print(f"updated {updated} reports, {failed} failed, under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
