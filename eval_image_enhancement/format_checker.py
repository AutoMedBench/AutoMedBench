"""Format validator for agent-submitted enhanced.npy files.

Canonical 2-layer: references live in data/<dataset>/private/<pid>/reference.npy.
Agent submissions live in <output_dir>/agents_outputs/<pid>/enhanced.npy.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def check_case(agent_case_dir: str, expected_shape: tuple[int, int]) -> tuple[bool, str]:
    path = os.path.join(agent_case_dir, "enhanced.npy")
    if not os.path.isfile(path):
        return False, "missing_enhanced_npy"
    try:
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        return False, f"unreadable: {type(exc).__name__}"
    if arr.ndim != 2:
        return False, f"wrong_ndim={arr.ndim}"
    if tuple(arr.shape) != tuple(expected_shape):
        return False, f"shape_mismatch: got {tuple(arr.shape)}, expected {tuple(expected_shape)}"
    if not np.issubdtype(arr.dtype, np.floating):
        return False, f"non_float_dtype={arr.dtype}"
    if np.isnan(arr).any() or np.isinf(arr).any():
        return False, "contains_nan_or_inf"
    return True, "ok"


def check_submission(agent_outputs_dir: str, private_dir: str, patient_ids: list[str]) -> dict[str, Any]:
    """Check every patient. Returns summary dict."""
    results: list[dict[str, Any]] = []
    all_valid = True
    for pid in patient_ids:
        ref_path = os.path.join(private_dir, pid, "reference.npy")
        if not os.path.isfile(ref_path):
            results.append({"patient_id": pid, "valid": False, "reason": "missing_reference"})
            all_valid = False
            continue
        ref = np.load(ref_path, mmap_mode="r")
        ok, reason = check_case(os.path.join(agent_outputs_dir, pid), tuple(ref.shape))
        results.append({"patient_id": pid, "valid": ok, "reason": reason})
        if not ok:
            all_valid = False

    n_valid = sum(1 for r in results if r["valid"])
    return {
        "output_format_valid": all_valid,
        "n_patients": len(patient_ids),
        "n_valid": n_valid,
        "completion_rate": n_valid / len(patient_ids) if patient_ids else 0.0,
        "per_patient": results,
    }
