#!/usr/bin/env python3
"""Failure classification for MedAgentsBench VQA runs."""

from __future__ import annotations

ERROR_CODES = {
    "E1": "Hallucination",
    "E2": "Resource error",
    "E3": "Logic error",
    "E4": "Code error",
    "E5": "Format error",
    "E8": "S3 artefacts never written",
}


def classify_failure(report: dict) -> dict | None:
    fmt = report.get("format", {})
    metrics = report.get("metrics", {})
    counts = metrics.get("counts", {})

    placeholder_rate = float(metrics.get("placeholder_rate", 0.0))
    model_call_detected = bool(metrics.get("model_call_detected", True))
    smoke_forward_passed = metrics.get("smoke_forward_passed")
    inference_mode = metrics.get("inference_mode", "real")
    postprocess_info = metrics.get("postprocess_info") or {}
    valid_outputs = counts.get("valid_outputs", 0)

    # BUG-044: S2 smoke passed but S3 artefacts never written and 0 answers.
    # Agent stalled in S3 debug loop and exited before producing required
    # files. Distinguish from generic empty / placeholder classifications.
    pp_path = postprocess_info.get("postprocess_path")
    cal_path = postprocess_info.get("calibration_path")
    import os as _os
    pp_missing = bool(pp_path) and not _os.path.isfile(pp_path)
    cal_missing = bool(cal_path) and not _os.path.isfile(cal_path)
    if (
        smoke_forward_passed is True
        and pp_missing
        and cal_missing
        and valid_outputs == 0
    ):
        return {
            "primary_failure": "E8",
            "failure_explanation": (
                "S3 artefacts never written: both answer_postprocess.py and "
                "s3_calibration.json are missing, and no per-question "
                "answer.json files were produced, despite smoke_forward.json "
                "passing S2. Agent stalled in S3 calibration and exited before "
                "committing to postprocess/calibration."
            ),
            "step_failures": {"s1": None, "s2": None, "s3": "E8", "s4": "E8", "s5": None},
        }

    if placeholder_rate > 0.5 or inference_mode in ("heuristic", "placeholder", "empty"):
        return {
            "primary_failure": "E5",
            "failure_explanation": (
                f"Fake or placeholder outputs detected (mode={inference_mode}, "
                f"placeholder_rate={placeholder_rate:.2f}). raw_model_output did not come "
                "from a real VLM forward pass."
            ),
            "step_failures": {"s1": None, "s2": None, "s3": "E5", "s4": "E5", "s5": "E5"},
        }

    if not model_call_detected:
        return {
            "primary_failure": "E5",
            "failure_explanation": (
                "No evidence of model.generate()/from_pretrained() in conversation "
                "trace; agent likely skipped real VLM inference."
            ),
            "step_failures": {"s1": None, "s2": "E5", "s3": None, "s4": "E5", "s5": None},
        }

    if smoke_forward_passed is False:
        return {
            "primary_failure": "E2",
            "failure_explanation": (
                "smoke_forward.json missing or invalid (missing/low wall_s/success=False); "
                "S2 smoke forward pass did not produce required artefact."
            ),
            "step_failures": {"s1": None, "s2": "E2", "s3": None, "s4": None, "s5": None},
        }

    postprocess_valid = metrics.get("postprocess_valid")
    if postprocess_valid is False:
        return {
            "primary_failure": "E3",
            "failure_explanation": (
                "answer_postprocess.py / s3_calibration.json missing or invalid; "
                "S3 calibration artefact contract not met (requires >=15 samples + "
                "importable postprocess(raw)->str)."
            ),
            "step_failures": {"s1": None, "s2": None, "s3": "E3", "s4": None, "s5": None},
        }

    if inference_mode == "real_but_broken":
        return {
            "primary_failure": "E3",
            "failure_explanation": (
                f"Model ran and produced outputs but answer quality collapsed "
                f"(accuracy={metrics.get('accuracy', 0.0):.3f}). Likely answer "
                "post-processing / prompt mismatch; no placeholder but EM+F1 near 0."
            ),
            "step_failures": {"s1": None, "s2": None, "s3": "E3", "s4": "E3", "s5": None},
        }

    if not fmt.get("output_format_valid", False):
        return {
            "primary_failure": "E5",
            "failure_explanation": "One or more answer.json records were missing or malformed.",
            "step_failures": {"s1": None, "s2": None, "s3": None, "s4": "E5", "s5": "E5"},
        }

    completion_rate = metrics.get("completion_rate", 0.0)
    if completion_rate < 0.5:
        return {
            "primary_failure": "E5",
            "failure_explanation": (
                f"Severe incompletion (completion_rate={completion_rate:.2f} < 0.5). "
                "Inference covered less than half the split."
            ),
            "step_failures": {"s1": None, "s2": None, "s3": None, "s4": "E5", "s5": None},
        }
    if completion_rate < 1.0:
        # Partial run: not a hard failure (rating may still be C/B/A by accuracy),
        # but flag so reports show it.
        return {
            "primary_failure": "E4",
            "failure_explanation": (
                f"Partial completion (completion_rate={completion_rate:.2f}); "
                "likely time_limit_s or unhandled per-sample error."
            ),
            "step_failures": {"s1": None, "s2": None, "s3": None, "s4": "E4", "s5": None},
        }

    if metrics.get("parse_rate", 0.0) < 1.0:
        return {
            "primary_failure": "E3",
            "failure_explanation": "Inference completed but some raw outputs could not be normalized to labels A-E.",
            "step_failures": {"s1": None, "s2": None, "s3": None, "s4": "E3", "s5": None},
        }

    if counts.get("valid_outputs", 0) <= 0:
        return {
            "primary_failure": "E5",
            "failure_explanation": "No valid output records were produced.",
            "step_failures": {"s1": None, "s2": None, "s3": None, "s4": None, "s5": "E5"},
        }

    if metrics.get("accuracy", 0.0) < 0.20:
        return {
            "primary_failure": "E3",
            "failure_explanation": "Outputs were valid but answer quality stayed near or below chance.",
            "step_failures": {"s1": None, "s2": None, "s3": None, "s4": "E3", "s5": None},
        }

    return None
