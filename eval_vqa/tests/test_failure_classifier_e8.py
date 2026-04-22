"""BUG-044: S3 artefacts never written → E8 classification."""

from __future__ import annotations

import os

from failure_classifier import classify_failure


def _base_report(
    pp_path: str,
    cal_path: str,
    *,
    smoke_passed: bool = True,
    valid_outputs: int = 0,
    completion_rate: float = 0.0,
) -> dict:
    return {
        "format": {"output_format_valid": False},
        "metrics": {
            "counts": {"valid_outputs": valid_outputs},
            "placeholder_rate": 0.0,
            "model_call_detected": True,
            "smoke_forward_passed": smoke_passed,
            "inference_mode": "s3_stalled" if smoke_passed else "empty",
            "postprocess_valid": False,
            "postprocess_info": {
                "postprocess_path": pp_path,
                "calibration_path": cal_path,
            },
            "completion_rate": completion_rate,
            "parse_rate": 0.0,
        },
    }


def test_e8_triggered_when_both_artefacts_missing(tmp_path):
    pp = str(tmp_path / "answer_postprocess.py")
    cal = str(tmp_path / "s3_calibration.json")
    # both files do NOT exist
    report = _base_report(pp, cal)
    result = classify_failure(report)
    assert result is not None
    assert result["primary_failure"] == "E8"
    assert "s3 artefacts" in result["failure_explanation"].lower() or "never" in result["failure_explanation"].lower()
    assert result["step_failures"]["s3"] == "E8"
    assert result["step_failures"]["s4"] == "E8"


def test_e8_not_triggered_when_smoke_failed(tmp_path):
    pp = str(tmp_path / "answer_postprocess.py")
    cal = str(tmp_path / "s3_calibration.json")
    report = _base_report(pp, cal, smoke_passed=False)
    result = classify_failure(report)
    # should fall through to E2 (smoke failure) instead
    assert result is not None
    assert result["primary_failure"] != "E8"


def test_e8_not_triggered_when_any_answer_written(tmp_path):
    pp = str(tmp_path / "answer_postprocess.py")
    cal = str(tmp_path / "s3_calibration.json")
    report = _base_report(pp, cal, valid_outputs=3, completion_rate=0.1)
    result = classify_failure(report)
    assert result is None or result["primary_failure"] != "E8"


def test_e8_not_triggered_when_postprocess_exists(tmp_path):
    pp = str(tmp_path / "answer_postprocess.py")
    cal = str(tmp_path / "s3_calibration.json")
    pp_open = open(pp, "w")
    pp_open.write("def postprocess(s): return s\n")
    pp_open.close()
    report = _base_report(pp, cal)
    # pp exists, cal missing → falls through E8 guard
    result = classify_failure(report)
    assert result is None or result["primary_failure"] != "E8"
