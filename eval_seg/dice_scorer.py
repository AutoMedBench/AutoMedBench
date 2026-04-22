#!/usr/bin/env python3
"""Compute Dice coefficient between prediction and ground truth masks."""

import numpy as np
import nibabel as nib


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice = 2|P ∩ G| / (|P| + |G|). Both empty → 1.0, one empty → 0.0."""
    pred_bin = (pred > 0.5).astype(bool)
    gt_bin = (gt > 0.5).astype(bool)

    sum_pred = pred_bin.sum()
    sum_gt = gt_bin.sum()

    if sum_pred == 0 and sum_gt == 0:
        return 1.0
    if sum_pred == 0 or sum_gt == 0:
        return 0.0

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    return float(2.0 * intersection / (sum_pred + sum_gt))


def score_patient(pred_path: str, gt_path: str) -> dict:
    """Score a single patient's mask against ground truth."""
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)

    pred_data = pred_img.get_fdata()
    gt_data = gt_img.get_fdata()

    if pred_data.shape != gt_data.shape:
        return {
            "dice": 0.0,
            "error": f"Shape mismatch: pred {pred_data.shape} vs gt {gt_data.shape}",
        }

    return {
        "dice": dice_coefficient(pred_data, gt_data),
        "pred_voxels": int((pred_data > 0.5).sum()),
        "gt_voxels": int((gt_data > 0.5).sum()),
    }


def _find_mask(directory: str, patterns: list) -> str | None:
    """Find first existing file matching any of the given patterns."""
    import os
    for pattern in patterns:
        path = os.path.join(directory, pattern)
        if os.path.isfile(path):
            return path
    return None


# Standard naming patterns per organ (checked in order)
ORGAN_PATTERNS = [
    "liver.nii.gz", "kidney.nii.gz", "pancreas.nii.gz", "organ.nii.gz",
]
LESION_PATTERNS = [
    "liver_lesion.nii.gz", "kidney_lesion.nii.gz", "pancreas_lesion.nii.gz",
    "lesion.nii.gz",
]


def score_all(pred_dir: str, gt_dir: str, patient_ids: list,
              organ_name: str = None) -> dict:
    """Score all patients. Returns per-patient and mean Dice.

    Lesion Dice is only averaged over patients where GT has foreground
    (gt_has_lesion=True), following the standard convention in medical
    segmentation benchmarks (MSD, KiTS). For GT-empty patients, false
    positives are penalized via Specificity, not Dice.

    Args:
        pred_dir: directory containing predicted masks per patient.
        gt_dir: directory containing ground-truth masks per patient.
        patient_ids: list of patient IDs (subdirectory names).
        organ_name: optional organ name (e.g. "liver") to look for
            ``{organ}.nii.gz`` and ``{organ}_lesion.nii.gz`` first.
    """
    import os

    # Build file-name search order (organ-specific first, generic fallback)
    if organ_name:
        organ_patterns = [f"{organ_name}.nii.gz"] + ORGAN_PATTERNS
        lesion_patterns = [f"{organ_name}_lesion.nii.gz"] + LESION_PATTERNS
    else:
        organ_patterns = ORGAN_PATTERNS
        lesion_patterns = LESION_PATTERNS

    results = {}
    lesion_dices = []       # only GT-positive patients
    lesion_dices_all = []   # all patients (for reference)
    organ_dices = []

    for pid in patient_ids:
        pred_pid_dir = os.path.join(pred_dir, pid)
        gt_pid_dir = os.path.join(gt_dir, pid)

        patient_result = {"lesion_dice": None, "organ_dice": None, "gt_has_lesion": False}

        # Lesion Dice
        pred_lesion = _find_mask(pred_pid_dir, lesion_patterns)
        gt_lesion = _find_mask(gt_pid_dir, lesion_patterns)

        if pred_lesion and gt_lesion:
            res = score_patient(pred_lesion, gt_lesion)
            patient_result["lesion_dice"] = res["dice"]
            patient_result["lesion_detail"] = res
            gt_has_lesion = res.get("gt_voxels", 0) > 0
            patient_result["gt_has_lesion"] = gt_has_lesion
            if "error" not in res:
                lesion_dices_all.append(res["dice"])
                # Only include in mean if GT actually has lesion
                if gt_has_lesion:
                    lesion_dices.append(res["dice"])

        # Organ Dice
        pred_organ = _find_mask(pred_pid_dir, organ_patterns)
        gt_organ = _find_mask(gt_pid_dir, organ_patterns)

        if pred_organ and gt_organ:
            res = score_patient(pred_organ, gt_organ)
            patient_result["organ_dice"] = res["dice"]
            patient_result["organ_detail"] = res
            if "error" not in res:
                organ_dices.append(res["dice"])

        results[pid] = patient_result

    mean_lesion_dice = float(np.mean(lesion_dices)) if lesion_dices else 0.0
    mean_organ_dice = float(np.mean(organ_dices)) if organ_dices else 0.0

    return {
        "per_patient": results,
        "mean_lesion_dice": mean_lesion_dice,
        "mean_lesion_dice_all": float(np.mean(lesion_dices_all)) if lesion_dices_all else 0.0,
        "mean_organ_dice": mean_organ_dice,
        "n_lesion_positive": len(lesion_dices),
        "n_patients": len(patient_ids),
    }


def score_all_multiclass(pred_dir: str, gt_dir: str, patient_ids: list,
                         tissue_labels: dict,
                         output_filename: str = "dseg.nii.gz") -> dict:
    """Multi-class Dice scoring for tasks with a single multi-label output.

    Both prediction and ground truth are expected at
    ``<dir>/<pid>/<output_filename>`` as single-volume label maps with
    integer values in ``{0} ∪ tissue_labels.keys()``. For each patient
    we compute a binary Dice per tissue label, and aggregate as:
      - per-patient mean Dice across all foreground tissues
      - dataset-level mean Dice per tissue class (macro-mean over patients)
      - overall macro-mean across tissues (the clinical score)

    Args:
        pred_dir: dir with predicted label maps per patient.
        gt_dir:   dir with ground-truth label maps per patient.
        patient_ids: list of patient IDs.
        tissue_labels: dict mapping integer label -> human tissue name,
            e.g. ``{1: "eCSF", 2: "GM", ..., 7: "BS"}``. Background (0)
            is excluded from the mean.
        output_filename: file name inside each patient dir (default
            ``dseg.nii.gz``).

    Returns:
        {
          "per_patient": {pid: {"per_class": {1: float, ...},
                                  "mean_tissue_dice": float,
                                  "missing_pred": bool,
                                  "missing_gt": bool}},
          "per_class_dice": {1: float, ...},   # macro-mean over patients
          "mean_tissue_dice_per_patient": [..], # per-patient macro-mean
          "macro_mean_dice": float,            # overall clinical score
          "tissue_labels": {1: "eCSF", ...},
          "n_patients": int,
        }
    """
    import os

    label_ids = sorted(int(k) for k in tissue_labels.keys())
    per_class_values = {lbl: [] for lbl in label_ids}
    per_patient = {}
    per_patient_means = []

    for pid in patient_ids:
        pred_path = os.path.join(pred_dir, pid, output_filename)
        gt_path = os.path.join(gt_dir, pid, output_filename)

        pr = {"per_class": {}, "mean_tissue_dice": None,
              "missing_pred": False, "missing_gt": False}

        if not os.path.isfile(pred_path):
            pr["missing_pred"] = True
            per_patient[pid] = pr
            continue
        if not os.path.isfile(gt_path):
            pr["missing_gt"] = True
            per_patient[pid] = pr
            continue

        try:
            pred_data = nib.load(pred_path).get_fdata()
            gt_data = nib.load(gt_path).get_fdata()
        except Exception as exc:
            pr["error"] = f"Failed to load: {exc}"
            per_patient[pid] = pr
            continue

        if pred_data.shape != gt_data.shape:
            pr["error"] = f"Shape mismatch: pred {pred_data.shape} vs gt {gt_data.shape}"
            for lbl in label_ids:
                pr["per_class"][lbl] = 0.0
                per_class_values[lbl].append(0.0)
            pr["mean_tissue_dice"] = 0.0
            per_patient_means.append(0.0)
            per_patient[pid] = pr
            continue

        pred_int = np.rint(pred_data).astype(np.int32)
        gt_int = np.rint(gt_data).astype(np.int32)

        tissue_dices = []
        for lbl in label_ids:
            d = dice_coefficient((pred_int == lbl).astype(np.uint8),
                                 (gt_int == lbl).astype(np.uint8))
            pr["per_class"][lbl] = d
            per_class_values[lbl].append(d)
            tissue_dices.append(d)
        pr["mean_tissue_dice"] = float(np.mean(tissue_dices))
        per_patient_means.append(pr["mean_tissue_dice"])
        per_patient[pid] = pr

    per_class_mean = {
        lbl: (float(np.mean(vals)) if vals else 0.0)
        for lbl, vals in per_class_values.items()
    }
    macro_mean = (float(np.mean(list(per_class_mean.values())))
                  if per_class_mean else 0.0)

    return {
        "per_patient": per_patient,
        "per_class_dice": per_class_mean,
        "mean_tissue_dice_per_patient": per_patient_means,
        "macro_mean_dice": macro_mean,
        "tissue_labels": {int(k): v for k, v in tissue_labels.items()},
        "n_patients": len(patient_ids),
    }
