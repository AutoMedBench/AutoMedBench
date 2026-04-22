#!/usr/bin/env python3
"""Aggregate scoring: S4, S5, Workflow, Task, Overall Score + Rating."""

# ---------------------------------------------------------------------------
# Step weights — used by both the per-step composite and the adaptive
# workflow score.  When S1-S3 are implemented, no code change is needed;
# just pass real values instead of None and the weights auto-normalise.
# ---------------------------------------------------------------------------
STEP_WEIGHTS = {
    "s1": 0.25, "s2": 0.15, "s3": 0.35, "s4": 0.15, "s5": 0.10,
}

# Rating is now assigned by assign_rating() based on medal tier + completion,
# not by numeric thresholds.




# ---------------------------------------------------------------------------
# Per-step scoring
# ---------------------------------------------------------------------------
def compute_s4(inference_completes: float, output_format_valid: float) -> float:
    """Step 4 score — inference completion + output format sanity."""
    return (0.50 * inference_completes +
            0.50 * output_format_valid)


def compute_s5(has_valid_results: bool, output_format_valid: bool) -> float:
    """Step 5 score — did agent produce valid, correctly formatted output?
    No clinical quality here — that's handled by clinical_score."""
    return (0.50 * float(has_valid_results) +
            0.50 * float(output_format_valid))


def compute_task_score(map_score: float) -> float:
    """Task score for 2D detection = mAP at the configured IoU threshold."""
    return round(map_score, 4)


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------
def compute_workflow_score(step_scores: dict, weights: dict = None) -> tuple:
    """Weighted average over all S1-S5 steps. Skipped steps count as 0.

    Args:
        step_scores: dict mapping step names to scores (or None).
        weights: optional per-step weights (defaults to STEP_WEIGHTS).

    Returns (score, list_of_completed_step_names).
    """
    w = weights or STEP_WEIGHTS
    completed = {k: v for k, v in step_scores.items() if v is not None}
    total_w = sum(w.values())
    score = sum(w[k] * (step_scores[k] or 0.0) for k in w) / total_w
    return round(score, 4), sorted(completed.keys())


def compute_overall_score(workflow: float, task_score: float) -> float:
    """Overall score = 0.50 * workflow + 0.50 * task score."""
    return round(0.50 * workflow + 0.50 * task_score, 4)


def assign_rating(overall: float, medal_tier: int = 0,
                   format_valid: bool = False) -> str:
    """Letter grade based on completion and quality tier.

    A — Good result (task score >= good threshold)
    B — Okay result (above baseline, below good)
    C — Below baseline
    F — Failed (invalid output or no output)
    """
    if not format_valid:
        return "F"
    if medal_tier >= 2:
        return "A"
    if medal_tier >= 1:
        return "B"
    return "C"


def is_resolved(rating: str) -> bool:
    """Resolved = Rating A or B (good or okay result)."""
    return rating in ("A", "B")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def build_report(format_result: dict, det_result: dict,
                 medal_result: dict, step_weights: dict = None) -> dict:
    """Build full evaluation report from component results."""
    output_valid = 1.0 if format_result["output_format_valid"] else 0.0

    medal = medal_result["tier"]
    task_score = det_result.get("mAP", 0.0)
    precision = det_result.get("micro_precision", 0.0)
    recall = det_result.get("micro_recall", 0.0)
    inference_completes = det_result.get("completion_rate", 0.0)
    total_patients = max(len(det_result.get("per_patient", {})), 1)

    # ---- Completeness gate ----
    all_patients_done = (inference_completes >= 1.0)
    if not all_patients_done:
        task_score = 0.0
        precision = 0.0
        recall = 0.0
        output_valid = 0.0
        medal = 0
        format_result = dict(format_result, output_format_valid=False)

    # ---- Per-step scores ----
    # S1-S3 = None (not yet evaluated); will become floats once implemented.
    s4 = compute_s4(inference_completes, output_valid)
    has_valid_results = inference_completes > 0 and task_score > 0
    s5 = compute_s5(has_valid_results, bool(output_valid))

    step_scores = {"s1": None, "s2": None, "s3": None, "s4": s4, "s5": s5}

    # ---- Aggregate scores ----
    workflow_score, active_steps = compute_workflow_score(step_scores,
                                                         weights=step_weights)
    benchmark_task_score = compute_task_score(task_score)
    overall_score = compute_overall_score(workflow_score, benchmark_task_score)
    preds_valid = format_result["output_format_valid"]
    rating = assign_rating(overall_score, medal_tier=medal,
                           format_valid=preds_valid)
    resolved = is_resolved(rating)

    # ---- Progress rate (active gates only) ----
    active_gates = 0
    passed_gates = 0
    # Gate s4: inference completeness >= 0.9
    active_gates += 1
    if inference_completes >= 0.9:
        passed_gates += 1
    # Gate s5: prediction format valid
    active_gates += 1
    if preds_valid:
        passed_gates += 1
    progress_rate = passed_gates / active_gates if active_gates > 0 else 0.0

    return {
        "step_scores": step_scores,
        "metrics": {
            "mAP": task_score,
            "micro_precision": precision,
            "micro_recall": recall,
            "per_class_ap": det_result.get("per_class_ap", {}),
            "total_tp": det_result.get("total_tp", 0),
            "total_fp": det_result.get("total_fp", 0),
            "total_fn": det_result.get("total_fn", 0),
            "iou_threshold": det_result.get("iou_threshold", 0.4),
            "medal_tier": medal,
            "medal_name": medal_result["name"],
        },
        "format": {
            "submission_format_valid": preds_valid,
            "output_format_valid": preds_valid,
        },
        "aggregate": {
            "rating": rating,
            "resolved": resolved,
            "overall_score": overall_score,
            "agentic_score": workflow_score,
            "clinical_score": benchmark_task_score,
            "progress_rate": progress_rate,
            "active_steps": active_steps,
        },
    }
