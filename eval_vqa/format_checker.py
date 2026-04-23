#!/usr/bin/env python3
"""Validate VQA output files against the benchmark schema."""

from __future__ import annotations

import json
import os
from typing import Any

from answer_utils import VALID_LABELS, normalize_options


REQUIRED_KEYS = {
    "question_id",
    "predicted_label",
    "predicted_answer",
    "raw_model_output",
    "model_name",
    "runtime_s",
}

PLACEHOLDER_PREFIXES = (
    "heuristic:",
    "fallback:",
    "placeholder:",
    "mock:",
    "dummy:",
    "stub:",
)
PLACEHOLDER_WHOLE = {"unknown", "n/a", "na", "none", "null"}


def detect_placeholder(raw_model_output: object, predicted_answer: object = None) -> tuple[bool, str]:
    """Flag fabricated / filler outputs.

    An honest skip (both ``raw_model_output`` and ``predicted_answer`` empty)
    is NOT a placeholder — the scorer simply drops it from ``valid_outputs``.
    Only fillers like ``unknown`` / ``heuristic:`` / ``fallback:`` are flagged,
    because they pretend to be an answer when no VLM pass occurred.
    """
    raw = (raw_model_output or "") if isinstance(raw_model_output, str) else ""
    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower()
    pa_stripped = predicted_answer.strip() if isinstance(predicted_answer, str) else ""
    pa_lower = pa_stripped.lower()

    # Honest skip: nothing claimed on either side.
    if not raw_stripped and not pa_stripped:
        return False, ""

    # Half-empty: non-empty answer without backing model text is a forged record.
    if not raw_stripped:
        return True, "raw_empty_with_answer"

    if raw_lower in PLACEHOLDER_WHOLE:
        return True, "raw_empty_or_generic"
    for prefix in PLACEHOLDER_PREFIXES:
        if raw_lower.startswith(prefix):
            return True, f"raw_prefix_{prefix.rstrip(':')}"
    return False, ""


def check_answer_file(
    answer_path: str,
    question: dict[str, Any],
    answer_mode: str = "multiple_choice",
) -> dict[str, Any]:
    result = {
        "exists": False,
        "valid": False,
        "parsed": False,
        "is_placeholder": False,
        "placeholder_reason": "",
        "errors": [],
        "record": None,
    }

    if not os.path.isfile(answer_path):
        result["errors"].append(f"answer.json not found: {answer_path}")
        return result
    result["exists"] = True

    try:
        with open(answer_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception as exc:
        result["errors"].append(f"JSON parse error: {exc}")
        return result

    result["record"] = record
    missing = REQUIRED_KEYS - set(record)
    if missing:
        result["errors"].append(f"Missing keys: {sorted(missing)}")
        return result

    if record.get("question_id") != question.get("question_id"):
        result["errors"].append("question_id mismatch")

    open_ended = answer_mode == "open_ended"
    label = str(record.get("predicted_label", "")).strip().upper()
    predicted_answer = record.get("predicted_answer")
    if open_ended:
        # Open-ended tasks (PathVQA, VQA-RAD) have no A-E label.
        # Parsed means a non-empty predicted_answer string.
        if isinstance(predicted_answer, str) and predicted_answer.strip():
            result["parsed"] = True
        else:
            result["errors"].append("predicted_answer must be a non-empty string for open-ended tasks")
    else:
        if label not in VALID_LABELS:
            result["errors"].append(f"Invalid predicted_label: {label!r}")
        else:
            result["parsed"] = True

    if not isinstance(record.get("raw_model_output"), str):
        result["errors"].append("raw_model_output must be a string")
    if not isinstance(record.get("model_name"), str) or not record.get("model_name"):
        result["errors"].append("model_name must be a non-empty string")
    if not isinstance(predicted_answer, str):
        result["errors"].append("predicted_answer must be a string")

    is_ph, reason = detect_placeholder(record.get("raw_model_output"), predicted_answer)
    if is_ph:
        result["is_placeholder"] = True
        result["placeholder_reason"] = reason
        result["parsed"] = False
        sample = str(record.get("raw_model_output") or "")[:60]
        result["errors"].append(f"raw_model_output placeholder ({reason}): {sample!r}")

    runtime_s = record.get("runtime_s")
    if not isinstance(runtime_s, (int, float)) or runtime_s < 0:
        result["errors"].append("runtime_s must be a non-negative number")

    if not open_ended:
        options = normalize_options(question.get("options", {}))
        if label in options and predicted_answer and predicted_answer != options[label]:
            result["errors"].append("predicted_answer does not match option text for predicted_label")

    if not result["errors"]:
        result["valid"] = True
    return result


def check_submission(
    agent_dir: str,
    question_ids: list[str],
    public_dir: str,
    answer_mode: str = "multiple_choice",
) -> dict[str, Any]:
    report = {
        "submission_format_valid": False,
        "output_format_valid": False,
        "per_question": {},
        "errors": [],
        "counts": {
            "expected": len(question_ids),
            "files_found": 0,
            "valid_files": 0,
            "parsed_files": 0,
            "placeholder_files": 0,
        },
        "placeholder_rate": 0.0,
        "placeholder_reasons": {},
    }

    all_valid = True
    for question_id in question_ids:
        question_path = os.path.join(public_dir, question_id, "question.json")
        with open(question_path, "r", encoding="utf-8") as handle:
            question = json.load(handle)

        answer_path = os.path.join(agent_dir, question_id, "answer.json")
        check = check_answer_file(answer_path, question, answer_mode=answer_mode)
        report["per_question"][question_id] = check
        if check["exists"]:
            report["counts"]["files_found"] += 1
        if check["valid"]:
            report["counts"]["valid_files"] += 1
        if check["parsed"]:
            report["counts"]["parsed_files"] += 1
        if check.get("is_placeholder"):
            report["counts"]["placeholder_files"] += 1
            reason = check.get("placeholder_reason") or "unknown"
            report["placeholder_reasons"][reason] = report["placeholder_reasons"].get(reason, 0) + 1
        if not check["valid"]:
            all_valid = False

    expected = max(report["counts"]["expected"], 1)
    report["placeholder_rate"] = round(report["counts"]["placeholder_files"] / expected, 4)
    # output_format_valid keeps all-or-nothing semantics (every file passes
    # schema checks). submission_format_valid is graded: >=50% valid answers
    # avoids hard-F when a handful of questions honestly return empty
    # (e.g. LLaVA-Med decode edge cases) while the rest are answered.
    valid_rate = report["counts"]["valid_files"] / expected
    report["valid_rate"] = round(valid_rate, 4)
    report["output_format_valid"] = all_valid
    report["submission_format_valid"] = valid_rate >= 0.5
    if not all_valid:
        for question_id, item in report["per_question"].items():
            for error in item["errors"]:
                report["errors"].append(f"{question_id}: {error}")
    return report
