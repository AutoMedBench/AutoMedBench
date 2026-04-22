#!/usr/bin/env python3
"""Output-format checks for report generation."""

from __future__ import annotations

import string
from pathlib import Path


PRINTABLE = set(string.printable) | {"\n", "\t"}


def check_report_file(
    path: str | Path,
    min_chars: int = 40,
    max_chars: int = 8000,
    min_alpha_chars: int = 20,
) -> dict:
    path = Path(path)
    result = {
        "path": str(path),
        "exists": path.is_file(),
        "valid": False,
        "char_count": 0,
        "alpha_char_count": 0,
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("missing_file")
        return result

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        result["errors"].append("utf8_decode_failed")
        return result

    result["char_count"] = len(text)
    result["alpha_char_count"] = sum(1 for char in text if char.isalpha())

    if result["char_count"] < min_chars:
        result["errors"].append("too_short")
    if result["char_count"] > max_chars:
        result["errors"].append("too_long")
    if result["alpha_char_count"] < min_alpha_chars:
        result["errors"].append("too_few_alpha_chars")
    if not text.strip():
        result["errors"].append("empty_after_strip")
    if any(char not in PRINTABLE for char in text):
        result["errors"].append("non_printable_characters")

    result["valid"] = not result["errors"]
    return result


def check_submission(agent_dir: str | Path, case_ids: list[str], task_config: dict) -> dict:
    agent_dir = Path(agent_dir)
    filename = task_config.get("output_filename", "report.txt")
    min_chars = int(task_config.get("report_min_chars", 40))
    max_chars = int(task_config.get("report_max_chars", 8000))
    min_alpha_chars = int(task_config.get("report_min_alpha_chars", 20))

    case_results: dict[str, dict] = {}
    valid_cases = []
    missing_cases = []
    invalid_cases = []

    for case_id in case_ids:
        report_path = agent_dir / case_id / filename
        result = check_report_file(
            report_path,
            min_chars=min_chars,
            max_chars=max_chars,
            min_alpha_chars=min_alpha_chars,
        )
        case_results[case_id] = result
        if not result["exists"]:
            missing_cases.append(case_id)
        if result["valid"]:
            valid_cases.append(case_id)
        else:
            invalid_cases.append(case_id)

    cases_total = len(case_ids)
    completion_rate = round(len(valid_cases) / cases_total, 4) if cases_total else 0.0
    output_format_valid = len(valid_cases) == cases_total and cases_total > 0

    return {
        "case_results": case_results,
        "cases_total": cases_total,
        "valid_cases": valid_cases,
        "invalid_cases": invalid_cases,
        "missing_cases": missing_cases,
        "cases_with_valid_output": len(valid_cases),
        "completion_rate": completion_rate,
        "output_format_valid": output_format_valid,
        "submission_format_valid": output_format_valid,
        "any_valid_results": bool(valid_cases),
    }
