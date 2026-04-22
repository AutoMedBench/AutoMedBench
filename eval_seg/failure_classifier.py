#!/usr/bin/env python3
"""Failure classification for MedAgentsBench.

Two dimensions:
  - Step failures (S1-S5): WHICH step failed
  - Error codes  (E1-E5): WHY it failed (root cause type)

Error Code Taxonomy:
  E1 — Hallucination:  Agent fabricates non-existent models, APIs, repos,
       packages, or functions that do not exist.
  E2 — Resource Error:  GPU OOM, execution timeout, download failure,
       disk space, network errors.
  E3 — Logic Error:    Code runs without crashing but produces incorrect
       results — wrong label mapping, orientation mismatch, bad
       preprocessing, incorrect thresholds, missing post-processing.
  E4 — Code Error:     Python/bash runtime errors — syntax errors, import
       errors, type errors, unhandled exceptions.
  E5 — Format Error:   Output does not meet the required spec — wrong
       shape, values not binary, missing files, malformed CSV.

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
    "E3": "Code ran but produced incorrect results (wrong labels, orientation, thresholds)",
    "E4": "Python/bash runtime error (syntax, import, type, exception)",
    "E5": "Output does not meet spec (wrong shape, not binary, missing files, bad CSV)",
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

    # Success override: if masks are valid and clinical quality is above
    # the baseline threshold, the run succeeded — do not flag failures.
    # This prevents false positives like S5:E5 on successful runs
    # caused by optional fields (e.g., missing decision CSV).
    masks_valid = fmt.get("output_format_valid", False)
    lesion_dice_check = metrics.get("lesion_dice", 0)
    medal = metrics.get("medal_tier", 0)
    if masks_valid and (medal >= 1 or lesion_dice_check >= 0.3):
        return None

    # Check gates in forward order (earliest root cause wins)

    # Output masks missing or invalid → S4 failed
    if not fmt.get("output_format_valid", False):
        return {
            "primary_failure": "E5",
            "failure_explanation": "Inference produced invalid or missing output masks.",
            "step_failures": {
                "s1": None, "s2": None, "s3": None,
                "s4": "E5", "s5": None,
            },
        }

    # Submission format invalid → S5 failed
    # Note: decision CSV is optional — only masks determine submission validity.
    if not fmt.get("submission_format_valid", False):
        reason = []
        if not fmt.get("output_format_valid", False):
            reason.append("Output masks invalid")
        csv_status = fmt.get("decision_csv_valid")
        if csv_status is False:  # explicitly False (present but malformed)
            reason.append("Decision CSV present but malformed")
        return {
            "primary_failure": "E5",
            "failure_explanation": "; ".join(reason) if reason else "Submission format check failed.",
            "step_failures": {
                "s1": None, "s2": None, "s3": None,
                "s4": None, "s5": "E5",
            },
        }

    # Format OK but clinical metrics too low → likely logic error
    lesion_dice = metrics.get("lesion_dice", 0)
    sensitivity = metrics.get("sensitivity")

    # Only flag E3 based on lesion Dice.  Sensitivity may be None when the
    # decision CSV was not generated (which is optional — agents are not
    # required to file a decision CSV).  Treating None as 0 caused false-
    # positive E3 flags on every run without a CSV (Bug 018).
    if lesion_dice < 0.1:
        sens_str = f"{sensitivity:.3f}" if sensitivity is not None else "N/A"
        return {
            "primary_failure": "E3",
            "failure_explanation": (
                f"Inference completed but output quality too low "
                f"(Dice={lesion_dice:.3f}, Sens={sens_str}). "
                f"Likely wrong label mapping or missing lesion detection."
            ),
            "step_failures": {
                "s1": None, "s2": None, "s3": None,
                "s4": "E3", "s5": None,
            },
        }

    # No failure detected
    return None
