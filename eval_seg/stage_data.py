#!/usr/bin/env python3
"""Stage Dataset701_CruzBench into MedAgentsBench benchmark layout for kidney and liver tasks."""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRUZBENCH = os.path.join(SCRIPT_DIR, "..", "..", "Data", "Dataset701_CruzBench")
OUT_BASE = os.path.join(SCRIPT_DIR, "data")

TASKS = {
    "kidney": {
        "organ_files": ["kidney_left.nii.gz", "kidney_right.nii.gz"],  # merge
        "lesion_file": "kidney_lesion.nii.gz",
        "label_col": "kidney_tumor",
    },
    "liver": {
        "organ_files": ["liver.nii.gz"],
        "lesion_file": "liver_lesion.nii.gz",
        "label_col": "liver_tumor",
    },
}


def stage_task(task_name: str, task_cfg: dict, patients: list, label_df: pd.DataFrame):
    print(f"\n--- Staging task: {task_name} ---")
    pub_dir = os.path.join(OUT_BASE, task_name, "public")
    priv_dir = os.path.join(OUT_BASE, task_name, "private")
    mask_dir = os.path.join(priv_dir, "masks")

    gt_rows = []

    for pid in patients:
        src = os.path.join(CRUZBENCH, pid)
        seg_dir = os.path.join(src, "segmentations")

        # Public: symlink CT
        ct_dst_dir = os.path.join(pub_dir, pid)
        os.makedirs(ct_dst_dir, exist_ok=True)
        ct_link = os.path.join(ct_dst_dir, "ct.nii.gz")
        if os.path.islink(ct_link) or os.path.isfile(ct_link):
            os.remove(ct_link)
        os.symlink(os.path.abspath(os.path.join(src, "ct.nii.gz")), ct_link)

        # Private: organ mask
        pat_mask_dir = os.path.join(mask_dir, pid)
        os.makedirs(pat_mask_dir, exist_ok=True)

        # Load and merge organ masks
        ref_img = None
        organ_data = None
        for organ_file in task_cfg["organ_files"]:
            fpath = os.path.join(seg_dir, organ_file)
            img = nib.load(fpath)
            d = (img.get_fdata() > 0.5).astype(np.uint8)
            if organ_data is None:
                organ_data = d
                ref_img = img
            else:
                organ_data = np.maximum(organ_data, d)

        organ_out = os.path.join(pat_mask_dir, "organ.nii.gz")
        nib.save(nib.Nifti1Image(organ_data, ref_img.affine, ref_img.header), organ_out)

        # Private: lesion mask
        les_img = nib.load(os.path.join(seg_dir, task_cfg["lesion_file"]))
        les_data = (les_img.get_fdata() > 0.5).astype(np.uint8)
        les_out = os.path.join(pat_mask_dir, "lesion.nii.gz")
        nib.save(nib.Nifti1Image(les_data, les_img.affine, les_img.header), les_out)

        # GT label
        row = label_df[label_df["patient_id"] == pid]
        lesion_present = int(row[task_cfg["label_col"]].values[0])
        gt_rows.append({"patient_id": pid, "organ": task_name, "lesion_present": lesion_present})

        organ_vox = int(organ_data.sum())
        les_vox = int(les_data.sum())
        print(f"  {pid}: organ={organ_vox:>8} vox, lesion={les_vox:>8} vox, label={lesion_present}")

    # Write ground_truth.csv
    gt_df = pd.DataFrame(gt_rows)
    gt_csv = os.path.join(priv_dir, "ground_truth.csv")
    gt_df.to_csv(gt_csv, index=False)
    print(f"  ground_truth.csv -> {gt_csv}")

    pos = sum(1 for r in gt_rows if r["lesion_present"] == 1)
    neg = len(gt_rows) - pos
    print(f"  Positive: {pos}, Negative: {neg}")


def main():
    if not os.path.isdir(CRUZBENCH):
        sys.exit(f"CruzBench not found: {CRUZBENCH}")

    label_df = pd.read_csv(os.path.join(CRUZBENCH, "label.csv"))
    patients = sorted(label_df["patient_id"].tolist())
    print(f"Found {len(patients)} patients: {patients}")

    for task_name, task_cfg in TASKS.items():
        stage_task(task_name, task_cfg, patients, label_df)

    print("\nStaging complete.")


if __name__ == "__main__":
    main()
