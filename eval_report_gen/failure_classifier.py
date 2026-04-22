#!/usr/bin/env python3
"""Heuristic failure classification for report generation."""

from __future__ import annotations


ERROR_CODES = {
    "E1": "Hallucination",
    "E2": "Resource error",
    "E3": "Logic error",
    "E4": "Code error",
    "E5": "Format error",
}


def classify_failure(report: dict) -> dict | None:
    fmt = report.get("format", {})
    metrics = report.get("metrics", {})

    if not fmt.get("output_format_valid", False):
        return {
            "primary_failure": "E5",
            "failure_explanation": "One or more study reports were missing or invalid.",
            "step_failures": {
                "s1": None,
                "s2": None,
                "s3": None,
                "s4": "E5",
                "s5": None,
            },
        }

    observation_f1 = metrics.get("observation_f1", 0.0)
    similarity = metrics.get("report_similarity", 0.0)

    if observation_f1 < 0.25 and similarity < 0.30:
        return {
            "primary_failure": "E3",
            "failure_explanation": (
                f"Reports were well-formed but clinically weak "
                f"(obs_f1={observation_f1:.3f}, sim={similarity:.3f})."
            ),
            "step_failures": {
                "s1": None,
                "s2": None,
                "s3": None,
                "s4": "E3",
                "s5": None,
            },
        }

    return None
