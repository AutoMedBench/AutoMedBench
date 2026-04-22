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
```bash
# For nnU-Net v2 CLI (use the config that matches YOUR model, e.g. 3d_lowres, 3d_fullres):
export CUDA_VISIBLE_DEVICES=0
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET -c CONFIG_NAME

# NEVER use map_location='cpu' or --disable_mixed_precision unless GPU
# is genuinely unavailable. NEVER patch torch.load to force CPU.
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
# Expected: a SINGLE multi-class label map with integer values in {0..7}.
dseg = ...   # numpy int array, same shape as scan_data

# Verify output
assert dseg.shape == scan_data.shape, \
    f"Shape mismatch: {dseg.shape} vs {scan_data.shape}"
unique_vals = set(np.unique(dseg).astype(int).tolist())
assert unique_vals.issubset({0, 1, 2, 3, 4, 5, 6, 7}), \
    f"Unexpected label values: {unique_vals}"

# Sanity-check per-tissue voxel counts — missing classes indicate a
# bad label mapping and will score 0 on that tissue.
TISSUE_NAMES = {{1: "eCSF", 2: "GM", 3: "WM", 4: "LV",
                5: "CBM", 6: "SGM", 7: "BS"}}
missing = []
for label, name in TISSUE_NAMES.items():
    count = int((dseg == label).sum())
    print(f"  label {label} ({name}): {count} voxels")
    if count == 0:
        missing.append(name)
if missing:
    print(f"WARNING: empty tissue labels: {missing}. "
          f"Go back to S1 and pick a model whose label scheme actually covers all 7 FeTA tissues.")

# Save as single multi-class label map
dseg_nii = nib.Nifti1Image(dseg.astype(np.uint8), scan_nii.affine, scan_nii.header)
nib.save(dseg_nii, "{output_dir}/agents_outputs/<first_patient_id>/{output_filename}")
print("Shape match: OK, labels in {0..7}: OK")
```
