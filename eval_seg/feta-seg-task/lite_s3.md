Skill — GPU setup (MANDATORY for S3 and S4, examples only — use any approach that works):
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
assert device.type == 'cuda', "GPU not available — check CUDA_VISIBLE_DEVICES"

# When loading checkpoints, ALWAYS use map_location='cuda':
checkpoint = torch.load(path, map_location='cuda', weights_only=False)
model = model.to(device)
```

Skill — How to validate multi-tissue output on one patient (examples only — use any approach that works):
```python
import nibabel as nib
import numpy as np

# Load input T2w super-resolution volume
scan_path = "{data_dir}/<first_patient_id>/{input_filename}"
scan_nii = nib.load(scan_path)
scan_data = scan_nii.get_fdata()
print(f"Scan shape: {scan_data.shape}, dtype: {scan_data.dtype}")

# Run inference (model-specific) — ON GPU
# ... your inference code here ...
# NOTE: many fetal-brain models resample to a fixed spacing (e.g. 0.5 mm
# isotropic) internally. You MUST resample the prediction back to the
# original scan geometry (shape, affine, spacing) before saving.
# Use scipy.ndimage.zoom with order=0 (nearest neighbour) for label maps,
# or monai.transforms.Invert / torchio.Resample.

# Expected: a SINGLE multi-class label map with integer values in {0..7}.
dseg = ...   # numpy int array, same shape as scan_data

# Verify output
assert dseg.shape == scan_data.shape, \
    f"Shape mismatch: got {dseg.shape}, expected {scan_data.shape}. Did you resample back?"
unique_vals = set(np.unique(dseg).astype(int).tolist())
assert unique_vals.issubset({0, 1, 2, 3, 4, 5, 6, 7}), \
    f"Unexpected label values: {unique_vals} — should be subset of 0..7"

# Sanity-check the per-tissue voxel counts. All 7 foreground classes
# should be non-empty for a healthy reconstruction; missing classes
# indicate a broken label scheme or a wrong output head.
TISSUE_NAMES = {{1: "eCSF", 2: "GM", 3: "WM", 4: "LV",
                5: "CBM", 6: "SGM", 7: "BS"}}
for label, name in TISSUE_NAMES.items():
    count = int((dseg == label).sum())
    print(f"  label {label} ({name}): {count} voxels")
    if count == 0:
        print(f"  WARNING: no voxels for {name} — check label mapping!")

# Save as a single multi-class label map
dseg_nii = nib.Nifti1Image(dseg.astype(np.uint8), scan_nii.affine, scan_nii.header)
nib.save(dseg_nii, "{output_dir}/agents_outputs/<first_patient_id>/{output_filename}")
print("Multi-tissue seg saved.")
```
