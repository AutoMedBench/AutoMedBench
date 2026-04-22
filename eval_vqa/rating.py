"""A/B/C/F rating helpers for scoring v0."""

from __future__ import annotations

from typing import Any


def rate_run(overall: float | int | None, validity: Any | None = None) -> str:
    """Return an A/B/C/F rating from overall score and validity.

    Rating assumptions:
        * Invalid runs are always rated F, even if the weighted score is high.
        * Valid runs receive A at 85+, B at 70+, C at 50+, and F below 50.
        * Dict validity is treated as valid unless it explicitly fails basic
          run-shape checks such as no samples, low format coverage, or very low
          completion coverage.
    """
    score = _safe_float(overall)
    if not is_valid_run(validity):
        return "F"
    if score >= 85.0:
        return "A"
    if score >= 70.0:
        return "B"
    if score >= 50.0:
        return "C"
    return "F"


def rating_report(overall: float | int | None, validity: Any | None = None) -> dict[str, Any]:
    """Return rating plus the validity decision used to compute it."""
    valid = is_valid_run(validity)
    return {
        "rating": rate_run(overall, validity),
        "overall": round(_safe_float(overall), 4),
        "valid": valid,
        "validity": validity if validity is not None else {"is_valid": True},
        "thresholds": {"A": 85.0, "B": 70.0, "C": 50.0, "F": 0.0},
    }


def assign_rating(overall: float | int | None, validity: Any | None = None) -> str:
    """Compatibility alias for `rate_run`."""
    return rate_run(overall, validity)


def is_valid_run(validity: Any | None = None) -> bool:
    """Evaluate whether a run is valid enough to receive A/B/C.

    Args:
        validity: `None`, bool, or a metrics dictionary. Supported dictionary
            fields include `is_valid`, `sample_count`, `completion_rate`, and
            `format_valid_coverage`.
    """
    if validity is None:
        return True
    if isinstance(validity, bool):
        return validity
    if not isinstance(validity, dict):
        return bool(validity)

    if "is_valid" in validity:
        return bool(validity["is_valid"])

    sample_count = _safe_float(validity.get("sample_count"))
    if sample_count <= 0:
        return False

    format_valid_coverage = _safe_float(validity.get("format_valid_coverage", 100.0))
    completion_rate = _safe_float(validity.get("completion_rate", 100.0))
    if format_valid_coverage < 80.0:
        return False
    if completion_rate < 50.0:
        return False
    return True


def _safe_float(value: Any) -> float:
    """Convert numeric-ish values to floats."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
