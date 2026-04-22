# How to Prepare Data

## Quick start (segmentation tasks — already scripted)

```bash
cd MedAgentsBench

# Stage all three organs (downloads ~50GB, takes ~1 hour)
python stage_data.py --organ all

# Or one at a time
python stage_data.py --organ kidney
python stage_data.py --organ liver
python stage_data.py --organ pancreas

# Keep temp files for debugging
python stage_data.py --organ kidney --no-cleanup
```

The script downloads from public sources (KiTS19, MSD Task03_Liver, PanTS), extracts binary organ + lesion masks, and writes everything to `data/`.

### Data sources

| Task | Source | Download | Label mapping |
|------|--------|----------|---------------|
| Kidney | KiTS19 (DigitalOcean + GitHub) | ~200MB per case, 20 cases | 0=bg, 1=kidney, 2=tumor |
| Liver | MSD Task03_Liver (HuggingFace) | ~30GB tar, 20 cases selected | 0=bg, 1=liver, 2=tumor |
| Pancreas | PanTS (HuggingFace BodyMaps/PanTSMini) | ~2GB per archive, 20 cases | 0=bg, 1=pancreas, 2=tumor |

All 60 cases are tumor-positive.

### Verify after staging

```bash
python -c "
from eval_seg.task_loader import discover_tasks, discover_patients
for tid in discover_tasks():
    pts = discover_patients(tid)
    print(f'{tid}: {len(pts)} patients')
"
```

Expected output:
```
kidney-seg-task: 20 patients
liver-seg-task: 40 patients
pancreas-seg-task: 20 patients
```

---

## Data layout (all domains)

Every dataset follows the same structure:

```
data/
  <DatasetName>/
    public/                        # Agent-visible input data
      <patient_id>/
        <input_file>               # One file per patient
      <patient_id>/
        <input_file>
      ...
    private/                       # Ground truth — agent NEVER sees this
      <patient_id>/
        <gt_files>                 # Domain-specific GT format
      <patient_id>/
        <gt_files>
      ...
      ground_truth.csv             # Summary: one row per patient
```

### Rules

| Rule | Why |
|------|-----|
| Agent only sees `public/` | Enforced by sandbox — accessing `private/` triggers violation |
| Patient IDs must match between `public/` and `private/` | The evaluator joins on patient ID |
| One `<input_file>` per patient folder in `public/` | The task config's `input_filename` tells `discover_patients()` what to look for |
| `ground_truth.csv` lives in `private/` | Summary for the evaluator, never shown to agent |

### Segmentation GT format

```
private/
  <patient_id>/
    organ.nii.gz                   # Binary mask (0/1), same shape as input scan
    lesion.nii.gz                  # Binary mask (0/1), same shape as input scan
  ground_truth.csv
```

`ground_truth.csv`:
```csv
patient_id,organ,lesion_present
Cruz_00000001,kidney,1
Cruz_00000002,kidney,0
```

### Other domain GT formats (examples)

**VQA:**
```
private/<patient_id>/answer.json       # {"answer": "yes"}
```

**Classification:**
```
private/<patient_id>/label.json        # {"label": "malignant"}
```

The GT format is whatever your domain's scorer expects. There is no enforced schema beyond: patient folders must exist, and `ground_truth.csv` should summarize them.

---

## How to add data for a new task

### Same domain (e.g., new organ for segmentation)

1. Create a new directory under `data/`:
   ```
   data/MyDataset_Spleen/
     public/
       PATIENT_001/ct.nii.gz
       PATIENT_002/ct.nii.gz
       ...
     private/
       PATIENT_001/organ.nii.gz
       PATIENT_001/lesion.nii.gz
       PATIENT_002/organ.nii.gz
       PATIENT_002/lesion.nii.gz
       ...
       ground_truth.csv
   ```

2. Set `data_dir_name: MyDataset_Spleen` in your task's `config.yaml`

3. Set `input_filename: ct.nii.gz` (or `t1.nii.gz`, etc.) to match your scan files

4. Verify:
   ```bash
   python -c "
   from eval_seg.task_loader import discover_patients
   print(discover_patients('spleen-seg-task'))
   "
   ```

### New domain (e.g., VQA)

Same layout, different GT files. Your domain's scorer decides what to look for in `private/`.

1. Create `data/<DatasetName>/public/` and `data/<DatasetName>/private/`
2. Put input data (images, questions) in `public/<patient_id>/`
3. Put answers / labels in `private/<patient_id>/`
4. Write `ground_truth.csv` in `private/`

---

## Writing a staging script

For reproducibility, automate your data preparation. See `stage_data.py` as a reference. Key steps:

1. **Download** from a public source (HuggingFace, DigitalOcean, Zenodo, etc.)
2. **Select cases** — filter for your criteria (e.g., tumor-positive only)
3. **Extract GT** — split multi-label segmentations into binary organ + lesion masks
4. **Assign patient IDs** — e.g., `Cruz_00000001`, `PAT_001`, etc.
5. **Write `ground_truth.csv`**
6. **Clean up** temp downloads

```python
# Minimal skeleton
def stage_my_task():
    for case in selected_cases:
        patient_id = f"PAT_{idx:06d}"

        # Public: copy input scan
        shutil.copy(case.image_path, f"data/MyDataset/public/{patient_id}/ct.nii.gz")

        # Private: extract binary masks from multi-label segmentation
        seg = nib.load(case.label_path).get_fdata()
        organ_mask = ((seg == ORGAN_LABEL) | (seg == TUMOR_LABEL)).astype(np.uint8)
        lesion_mask = (seg == TUMOR_LABEL).astype(np.uint8)
        nib.save(nib.Nifti1Image(organ_mask, affine), f"data/MyDataset/private/{patient_id}/organ.nii.gz")
        nib.save(nib.Nifti1Image(lesion_mask, affine), f"data/MyDataset/private/{patient_id}/lesion.nii.gz")

        gt_rows.append(f"{patient_id},myorgan,{int(lesion_mask.sum() > 0)}")

    # Write ground_truth.csv
    with open("data/MyDataset/private/ground_truth.csv", "w") as f:
        f.write("patient_id,organ,lesion_present\n")
        f.write("\n".join(gt_rows) + "\n")
```

---

## Checklist

- [ ] `data/<DatasetName>/public/<patient>/` has input files
- [ ] `data/<DatasetName>/private/<patient>/` has GT files
- [ ] `data/<DatasetName>/private/ground_truth.csv` exists with all patient IDs
- [ ] Patient IDs match between `public/` and `private/`
- [ ] `input_filename` in task `config.yaml` matches actual filenames in `public/`
- [ ] `data_dir_name` in task `config.yaml` matches the folder name under `data/`
- [ ] `discover_patients()` returns the expected patient list
- [ ] `data/` and `*.nii.gz` are in `.gitignore` (large files must not be committed)
