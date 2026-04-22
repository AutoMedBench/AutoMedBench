#!/usr/bin/env python3
"""Aggregate scoring: S4, S5, Workflow, Clinical, Overall Score + Rating."""

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


def compute_clinical_score(organ_dice: float, lesion_dice: float) -> float:
    """Clinical score = 0.50 * lesion Dice + 0.50 * organ Dice.

    Organ Dice: ALL patients.  Lesion Dice: positive patients only.
    """
    return round(0.50 * lesion_dice + 0.50 * organ_dice, 4)


def compute_clinical_score_multiclass(macro_mean_dice: float) -> float:
    """Multi-class clinical score = macro-mean Dice across foreground tissues."""
    return round(float(macro_mean_dice), 4)


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


def compute_overall_score(workflow: float, clinical: float) -> float:
    """Overall score = 0.50 * agentic (workflow) + 0.50 * clinical."""
    return round(0.50 * workflow + 0.50 * clinical, 4)


def assign_rating(overall: float, medal_tier: int = 0,
                   format_valid: bool = False) -> str:
    """Letter grade based on completion and quality tier.

    A — Good result (Dice >= good threshold)
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
def build_report(format_result: dict, dice_result: dict,
                 medal_result: dict, step_weights: dict = None,
                 task_cfg: dict = None) -> dict:
    """Build full evaluation report from component results.

    If ``task_cfg['task_type'] == 'multiclass'``, the clinical metrics
    come from ``dice_result['macro_mean_dice']`` and per-class dice;
    otherwise the classic organ/lesion binary scoring is used.
    """
    task_cfg = task_cfg or {}
    if task_cfg.get("task_type") == "multiclass":
        return _build_report_multiclass(format_result, dice_result,
                                        medal_result, step_weights, task_cfg)

    output_valid = 1.0 if format_result["output_format_valid"] else 0.0

    # mean_lesion_dice from dice_scorer is already positive-patient-only
    lesion_dice = dice_result.get("mean_lesion_dice", 0.0)

    # Organ dice on ALL patients (for metrics)
    organ_dices_all = [
        pr["organ_dice"]
        for pr in dice_result.get("per_patient", {}).values()
        if pr.get("organ_dice") is not None
    ]
    organ_dice = float(sum(organ_dices_all) / len(organ_dices_all)) if organ_dices_all else 0.0
    has_organ = len(organ_dices_all) > 0

    # Organ dice on POSITIVE patients only (for clinical score)
    pos_organ_dices = [
        pr["organ_dice"]
        for pr in dice_result.get("per_patient", {}).values()
        if pr.get("gt_has_lesion") and pr.get("organ_dice") is not None
    ]
    pos_organ_dice = float(sum(pos_organ_dices) / len(pos_organ_dices)) if pos_organ_dices else organ_dice

    medal = medal_result["tier"]

    # Inference completeness: fraction of patients with output files
    total_patients = max(len(dice_result.get("per_patient", {})), 1)
    patients_with_output = sum(
        1 for pr in dice_result.get("per_patient", {}).values()
        if pr.get("lesion_dice") is not None
    )
    inference_completes = patients_with_output / total_patients

    # ---- Completeness gate ----
    # Agents MUST process ALL patients. Partial completion = F with no
    # Dice credit.  This prevents time-limited runs from getting inflated
    # scores based on a subset of easy patients.
    all_patients_done = (inference_completes >= 1.0)
    if not all_patients_done:
        lesion_dice = 0.0
        organ_dice = 0.0
        pos_organ_dice = 0.0
        has_organ = False
        output_valid = 0.0
        medal = 0
        # Force masks_valid=False so rating becomes F
        format_result = dict(format_result, output_format_valid=False)

    # ---- Per-step scores ----
    # S1-S3 = None (not yet evaluated); will become floats once implemented.
    s4 = compute_s4(inference_completes, output_valid)
    has_valid_results = inference_completes > 0 and (lesion_dice > 0 or organ_dice > 0)
    s5 = compute_s5(has_valid_results, bool(output_valid))

    step_scores = {"s1": None, "s2": None, "s3": None, "s4": s4, "s5": s5}

    # ---- Aggregate scores ----
    workflow_score, active_steps = compute_workflow_score(step_scores,
                                                         weights=step_weights)
    # Organ Dice: ALL patients.  Lesion Dice: positive patients only.
    clinical_score = compute_clinical_score(organ_dice, lesion_dice)
    overall_score = compute_overall_score(workflow_score, clinical_score)
    masks_valid = format_result["output_format_valid"]
    rating = assign_rating(overall_score, medal_tier=medal,
                           format_valid=masks_valid)
    resolved = is_resolved(rating)

    # ---- Progress rate (active gates only) ----
    active_gates = 0
    passed_gates = 0
    # Gate s4: inference completeness >= 0.9
    active_gates += 1
    if inference_completes >= 0.9:
        passed_gates += 1
    # Gate s5: masks format valid
    active_gates += 1
    if masks_valid:
        passed_gates += 1
    progress_rate = passed_gates / active_gates if active_gates > 0 else 0.0

    return {
        "step_scores": step_scores,
        "metrics": {
            "lesion_dice": lesion_dice,
            "organ_dice": organ_dice if has_organ else "N/A",
            "pos_organ_dice": pos_organ_dice if has_organ else "N/A",
            "medal_tier": medal,
            "medal_name": medal_result["name"],
        },
        "format": {
            "submission_format_valid": masks_valid,
            "output_format_valid": masks_valid,
        },
        "aggregate": {
            "rating": rating,
            "resolved": resolved,
            "overall_score": overall_score,
            "agentic_score": workflow_score,
            "clinical_score": clinical_score,
            "progress_rate": progress_rate,
            "active_steps": active_steps,
        },
    }


def _build_report_multiclass(format_result: dict, dice_result: dict,
                             medal_result: dict, step_weights: dict,
                             task_cfg: dict) -> dict:
    """Build report for multi-class segmentation tasks (e.g. FeTA).

    Clinical score = macro-mean Dice across foreground tissues.
    No organ/lesion split and no patient-level TP/TN/FP/FN.
    """
    output_valid = 1.0 if format_result["output_format_valid"] else 0.0

    macro_mean_dice = float(dice_result.get("macro_mean_dice", 0.0))
    per_class_dice = {int(k): float(v)
                      for k, v in dice_result.get("per_class_dice", {}).items()}
    tissue_labels = {int(k): str(v)
                     for k, v in dice_result.get("tissue_labels", {}).items()}

    # Inference completeness: fraction of patients with a valid prediction file
    per_patient = dice_result.get("per_patient", {})
    total_patients = max(len(per_patient), 1)
    patients_with_output = sum(
        1 for pr in per_patient.values()
        if not pr.get("missing_pred") and not pr.get("missing_gt")
        and pr.get("mean_tissue_dice") is not None
    )
    inference_completes = patients_with_output / total_patients

    all_patients_done = (inference_completes >= 1.0)
    medal = medal_result["tier"]
    if not all_patients_done:
        macro_mean_dice = 0.0
        per_class_dice = {lbl: 0.0 for lbl in per_class_dice}
        output_valid = 0.0
        medal = 0
        format_result = dict(format_result, output_format_valid=False)

    # Per-step scores
    s4 = compute_s4(inference_completes, output_valid)
    has_valid_results = inference_completes > 0 and macro_mean_dice > 0
    s5 = compute_s5(has_valid_results, bool(output_valid))
    step_scores = {"s1": None, "s2": None, "s3": None, "s4": s4, "s5": s5}

    workflow_score, active_steps = compute_workflow_score(step_scores,
                                                         weights=step_weights)
    clinical_score = compute_clinical_score_multiclass(macro_mean_dice)
    overall_score = compute_overall_score(workflow_score, clinical_score)
    masks_valid = format_result["output_format_valid"]
    rating = assign_rating(overall_score, medal_tier=medal,
                           format_valid=masks_valid)
    resolved = is_resolved(rating)

    # Progress rate gates
    active_gates = 2
    passed_gates = 0
    if inference_completes >= 0.9:
        passed_gates += 1
    if masks_valid:
        passed_gates += 1
    progress_rate = passed_gates / active_gates

    # Human-readable per-tissue dice (tissue name keys)
    per_tissue_dice = {
        tissue_labels.get(lbl, str(lbl)): round(v, 4)
        for lbl, v in sorted(per_class_dice.items())
    }

    return {
        "step_scores": step_scores,
        "metrics": {
            "task_type": "multiclass",
            "macro_mean_dice": round(macro_mean_dice, 4),
            "per_class_dice": per_tissue_dice,
            "tissue_labels": tissue_labels,
            "medal_tier": medal,
            "medal_name": medal_result["name"],
        },
        "format": {
            "submission_format_valid": masks_valid,
            "output_format_valid": masks_valid,
        },
        "aggregate": {
            "rating": rating,
            "resolved": resolved,
            "overall_score": overall_score,
            "agentic_score": workflow_score,
            "clinical_score": clinical_score,
            "progress_rate": progress_rate,
            "active_steps": active_steps,
        },
    }
