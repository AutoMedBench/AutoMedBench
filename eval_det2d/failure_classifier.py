#!/usr/bin/env python3
"""Failure classification for MedAgentsBench 2D detection.

Two dimensions:
  - Step failures (S1-S5): WHICH step failed
  - Error codes  (E1-E5): WHY it failed (root cause type)

Error Code Taxonomy:
  E1 — Hallucination:  Agent fabricates non-existent models, APIs, repos,
       packages, or functions that do not exist.
  E2 — Resource Error:  GPU OOM, execution timeout, download failure,
       disk space, network errors.
  E3 — Logic Error:    Code runs without crashing but produces incorrect
       results — wrong label mapping, coordinate transforms, bad
       preprocessing, incorrect thresholds, or empty outputs.
  E4 — Code Error:     Python/bash runtime errors — syntax errors, import
       errors, type errors, unhandled exceptions.
  E5 — Format Error:   Output does not meet the required spec — missing
       JSON, malformed boxes, invalid coordinates, or missing files.

A failure is reported as: step + error code, e.g. "S3: E2" means
step 3 (Validate) failed due to a code error.
"""

ERROR_CODES = {
    "E1": "Hallucination",
    "E2": "Resource error",
    "E3": "Logic error",
    "E4": "Code error",
    "E5": "Format error",
}

ERROR_CODE_DESCRIPTIONS = {
    "E1": "Agent fabricated non-existent models, APIs, repos, or functions",
    "E2": "GPU OOM, timeout, download failure, or network error",
    "E3": "Code ran but produced incorrect boxes (wrong labels, transforms, thresholds)",
    "E4": "Python/bash runtime error (syntax, import, type, exception)",
    "E5": "Output does not meet spec (missing JSON, bad coordinates, malformed files)",
}


def classify_failure(report: dict) -> dict:
    """Auto-classify failure from an evaluation report.

    This is a lightweight heuristic fallback. The LLM judge provides
    more accurate per-step failure analysis when available.

    Returns dict with primary_failure, failure_explanation, step_failures,
    or None if no failure detected.
    """
    fmt = report.get("format", {})
    metrics = report.get("metrics", {})

    # Success override: if predictions are valid and task quality is above
    # the baseline threshold, the run succeeded — do not flag failures.
    preds_valid = fmt.get("output_format_valid", False)
    task_score = metrics.get("mAP", 0)
    medal = metrics.get("medal_tier", 0)
    if preds_valid and (medal >= 1 or task_score >= 0.3):
        return None

    # Check gates in forward order (earliest root cause wins)

    # Output JSON missing or invalid → S4 failed
    if not fmt.get("output_format_valid", False):
        return {
            "primary_failure": "E5",
            "failure_explanation": "Inference produced invalid or missing prediction.json files.",
            "step_failures": {
                "s1": None, "s2": None, "s3": None,
                "s4": "E5", "s5": None,
            },
        }

    # Submission format invalid → S5 failed
    # Only prediction JSON files determine submission validity.
    if not fmt.get("submission_format_valid", False):
        reason = []
        if not fmt.get("output_format_valid", False):
            reason.append("Prediction JSON invalid")
        return {
            "primary_failure": "E5",
            "failure_explanation": "; ".join(reason) if reason else "Submission format check failed.",
            "step_failures": {
                "s1": None, "s2": None, "s3": None,
                "s4": None, "s5": "E5",
            },
        }

    # Format OK but detection metrics too low → likely logic error
    map_score = metrics.get("mAP", 0)
    precision = metrics.get("micro_precision", 0)
    recall = metrics.get("micro_recall", 0)
    if map_score < 0.1:
        return {
            "primary_failure": "E3",
            "failure_explanation": (
                f"Inference completed but output quality too low "
                f"(mAP={map_score:.3f}, P={precision:.3f}, R={recall:.3f}). "
                f"Likely wrong coordinates, wrong preprocessing, or empty predictions."
            ),
            "step_failures": {
                "s1": None, "s2": None, "s3": None,
                "s4": "E3", "s5": None,
            },
        }

    # No failure detected
    return None
