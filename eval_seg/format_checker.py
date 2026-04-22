#!/usr/bin/env python3
"""Validate agent submission format against benchmark spec."""

import os
import numpy as np
import nibabel as nib
import pandas as pd


def check_decision_csv(csv_path: str, patient_ids: list) -> dict:
    """Validate agents_decision.csv."""
    result = {"exists": False, "valid": False, "errors": []}

    if not os.path.isfile(csv_path):
        result["errors"].append("agents_decision.csv not found")
        return result
    result["exists"] = True

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["errors"].append(f"CSV parse error: {e}")
        return result

    required_cols = {"patient_id", "organ", "lesion_present"}
    missing = required_cols - set(df.columns)
    if missing:
        result["errors"].append(f"Missing columns: {missing}")
        return result

    if df.isnull().any().any():
        result["errors"].append("CSV contains missing values")

    invalid_labels = df[~df["lesion_present"].isin([0, 1])]
    if len(invalid_labels) > 0:
        result["errors"].append(f"Non-binary lesion_present values: {invalid_labels['lesion_present'].tolist()}")

    submitted_ids = set(df["patient_id"].tolist())
    missing_ids = set(patient_ids) - submitted_ids
    if missing_ids:
        result["errors"].append(f"Missing patient IDs: {missing_ids}")

    if not result["errors"]:
        result["valid"] = True
    return result


def check_mask_file(mask_path: str, ref_ct_path: str = None) -> dict:
    """Validate a single mask file."""
    result = {"exists": False, "valid": False, "errors": []}

    if not os.path.isfile(mask_path):
        result["errors"].append(f"Mask not found: {mask_path}")
        return result
    result["exists"] = True

    try:
        img = nib.load(mask_path)
        data = img.get_fdata()
    except Exception as e:
        result["errors"].append(f"Failed to load mask: {e}")
        return result

    # Check binary values
    unique_vals = np.unique(data)
    if not all(v in [0.0, 1.0] for v in unique_vals):
        result["errors"].append(f"Non-binary values: {unique_vals}")

    # Check shape matches CT if provided
    if ref_ct_path and os.path.isfile(ref_ct_path):
        ref_img = nib.load(ref_ct_path)
        if data.shape != ref_img.shape:
            result["errors"].append(
                f"Shape mismatch: mask {data.shape} vs CT {ref_img.shape}"
            )

    if not result["errors"]:
        result["valid"] = True
    return result


def check_multiclass_mask_file(mask_path: str, ref_scan_path: str,
                               allowed_labels: set) -> dict:
    """Validate a single multi-class label-map file."""
    result = {"exists": False, "valid": False, "errors": []}

    if not os.path.isfile(mask_path):
        result["errors"].append(f"Mask not found: {mask_path}")
        return result
    result["exists"] = True

    try:
        img = nib.load(mask_path)
        data = img.get_fdata()
    except Exception as e:
        result["errors"].append(f"Failed to load mask: {e}")
        return result

    # Values must be integers within allowed_labels (0 always allowed)
    unique_vals = set(np.rint(np.unique(data)).astype(int).tolist())
    allowed = set(allowed_labels) | {0}
    unexpected = unique_vals - allowed
    if unexpected:
        result["errors"].append(f"Unexpected label values: {sorted(unexpected)} "
                                f"(allowed: {sorted(allowed)})")

    # Shape must match input scan
    if ref_scan_path and os.path.isfile(ref_scan_path):
        ref_img = nib.load(ref_scan_path)
        if data.shape != ref_img.shape:
            result["errors"].append(
                f"Shape mismatch: mask {data.shape} vs scan {ref_img.shape}"
            )

    if not result["errors"]:
        result["valid"] = True
    return result


def check_submission(agent_dir: str, patient_ids: list,
                     public_dir: str = None,
                     task_cfg: dict = None) -> dict:
    """Full submission format check.

    For binary tasks (default): checks ``<pid>/lesion.nii.gz`` (+ optional
    ``organ.nii.gz``) as binary masks.

    For multi-class tasks (``task_cfg['task_type'] == 'multiclass'``):
    checks a single label-map file per patient with integer values in
    ``{0} ∪ tissue_labels.keys()``.

    The decision CSV (agents_decision.csv) is optional and not applicable
    for multi-class tasks.
    """
    report = {
        "submission_format_valid": False,
        "output_format_valid": False,
        "decision_csv_valid": None,
        "per_patient": {},
        "errors": [],
    }

    task_cfg = task_cfg or {}
    is_multiclass = task_cfg.get("task_type") == "multiclass"
    input_filename = task_cfg.get("input_filename", "ct.nii.gz")

    all_masks_valid = True

    if is_multiclass:
        output_filename = task_cfg.get("output_filename", "dseg.nii.gz")
        allowed_labels = set(int(k) for k in (task_cfg.get("tissue_labels") or {}).keys())
        for pid in patient_ids:
            ref_scan = os.path.join(public_dir, pid, input_filename) if public_dir else None
            pred_path = os.path.join(agent_dir, pid, output_filename)
            mc_check = check_multiclass_mask_file(pred_path, ref_scan, allowed_labels)
            report["per_patient"][pid] = {"multiclass": mc_check}
            if not mc_check["valid"]:
                all_masks_valid = False
    else:
        # Binary organ + lesion (existing behavior)
        for pid in patient_ids:
            patient_report = {}
            lesion_path = os.path.join(agent_dir, pid, "lesion.nii.gz")
            ref_ct = os.path.join(public_dir, pid, input_filename) if public_dir else None

            lesion_check = check_mask_file(lesion_path, ref_ct)
            patient_report["lesion"] = lesion_check
            if not lesion_check["valid"]:
                all_masks_valid = False

            # Organ mask is optional for now
            organ_path = os.path.join(agent_dir, pid, "organ.nii.gz")
            if os.path.isfile(organ_path):
                organ_check = check_mask_file(organ_path, ref_ct)
                patient_report["organ"] = organ_check

            report["per_patient"][pid] = patient_report

    report["output_format_valid"] = all_masks_valid
    report["submission_format_valid"] = all_masks_valid

    # Optional: check decision CSV if present (binary tasks only)
    if not is_multiclass:
        csv_path = os.path.join(agent_dir, "agents_decision.csv")
        if os.path.isfile(csv_path):
            csv_result = check_decision_csv(csv_path, patient_ids)
            report["decision_csv_valid"] = csv_result["valid"]
        # else: decision_csv_valid stays None (not evaluated)

    return report
