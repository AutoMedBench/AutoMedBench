#!/usr/bin/env python3
"""Generate 3 dummy agent submissions for kidney and liver tasks."""

import os
import numpy as np
import nibabel as nib
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DUMMY_DIR = os.path.join(SCRIPT_DIR, "dummy_agents")

TASKS = ["kidney", "liver"]


def load_mask(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def save_mask(data, affine, header, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    nib.save(img, out_path)


def erode_mask(data):
    """Simple 6-connected erosion using numpy."""
    eroded = data.copy()
    padded = np.pad(eroded, 1, mode="constant", constant_values=0)
    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        shifted = padded[1+dx:1+dx+eroded.shape[0],
                         1+dy:1+dy+eroded.shape[1],
                         1+dz:1+dz+eroded.shape[2]]
        eroded = eroded * (shifted > 0.5).astype(eroded.dtype)
    return eroded


def make_dummy(task, patients, gt_df, dummy_name, mask_fn, decision_fn):
    """Generate one dummy agent's outputs for a task."""
    out_base = os.path.join(DUMMY_DIR, dummy_name, task)
    agent_dir = os.path.join(out_base, "agents_outputs")

    rows = []
    for pid in patients:
        gt_mask_dir = os.path.join(DATA_DIR, task, "private", "masks", pid)

        # Organ
        organ_data, aff, hdr = load_mask(os.path.join(gt_mask_dir, "organ.nii.gz"))
        organ_pred = mask_fn(organ_data)
        save_mask(organ_pred, aff, hdr, os.path.join(agent_dir, pid, "organ.nii.gz"))

        # Lesion
        les_data, aff, hdr = load_mask(os.path.join(gt_mask_dir, "lesion.nii.gz"))
        les_pred = mask_fn(les_data)
        save_mask(les_pred, aff, hdr, os.path.join(agent_dir, pid, "lesion.nii.gz"))

        # Decision
        gt_row = gt_df[gt_df["patient_id"] == pid].iloc[0]
        decision = decision_fn(gt_row["lesion_present"])
        rows.append({"patient_id": pid, "organ": task, "lesion_present": decision})

    csv_path = os.path.join(out_base, "agents_decision.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    for task in TASKS:
        print(f"\n--- Generating dummies for: {task} ---")
        gt_csv = os.path.join(DATA_DIR, task, "private", "ground_truth.csv")
        gt_df = pd.read_csv(gt_csv)
        patients = sorted(gt_df["patient_id"].tolist())

        # Perfect: copy GT exactly, correct decisions
        make_dummy(task, patients, gt_df,
                   "perfect",
                   mask_fn=lambda d: (d > 0.5).astype(np.uint8),
                   decision_fn=lambda lp: int(lp))
        print(f"  [perfect] done")

        # Partial: erode masks, correct decisions
        make_dummy(task, patients, gt_df,
                   "partial",
                   mask_fn=lambda d: erode_mask((d > 0.5).astype(np.uint8)),
                   decision_fn=lambda lp: int(lp))
        print(f"  [partial] done")

        # Empty: all zeros, all predict no lesion
        make_dummy(task, patients, gt_df,
                   "empty",
                   mask_fn=lambda d: np.zeros_like(d, dtype=np.uint8),
                   decision_fn=lambda lp: 0)
        print(f"  [empty] done")

    print("\nDummy agent generation complete.")


if __name__ == "__main__":
    main()
