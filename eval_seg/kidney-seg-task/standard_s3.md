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

Skill — How to validate on one patient (examples only — use any approach that works):
```python
import nibabel as nib
import numpy as np

# Load input scan (use the first patient discovered in the data directory)
scan_path = "{data_dir}/<first_patient_id>/{input_filename}"
scan_nii = nib.load(scan_path)
scan_data = scan_nii.get_fdata()
print(f"Scan shape: {scan_data.shape}, dtype: {scan_data.dtype}")

# Run inference (model-specific) — ON GPU
# ... your inference code here ...
organ_mask = ...   # numpy array, same shape as scan_data
lesion_mask = ...  # numpy array, same shape as scan_data

# Verify output
assert organ_mask.shape == scan_data.shape, f"Shape mismatch: {organ_mask.shape} vs {scan_data.shape}"
assert set(np.unique(organ_mask)).issubset({0, 1}), f"Not binary: {np.unique(organ_mask)}"
assert set(np.unique(lesion_mask)).issubset({0, 1}), f"Not binary: {np.unique(lesion_mask)}"
assert organ_mask.sum() > 1000, f"Organ mask too small: {organ_mask.sum()} voxels"

# CRITICAL: Check lesion output — ratio of 0.0 means model has no tumor labels
lesion_ratio = lesion_mask.sum() / max(organ_mask.sum(), 1)
print(f"Organ voxels: {organ_mask.sum()}, Lesion voxels: {lesion_mask.sum()}")
print(f"Lesion/Organ ratio: {lesion_ratio:.4f}")
if lesion_ratio == 0.0:
    print("WARNING: Lesion mask is EMPTY. Model is NOT detecting tumors.")
    print("Go back to S1 and choose a model with actual TUMOR labels.")
else:
    print(f"Lesion check: OK (ratio={lesion_ratio:.4f})")

# Save
organ_nii = nib.Nifti1Image(organ_mask.astype(np.uint8), scan_nii.affine, scan_nii.header)
nib.save(organ_nii, "{output_dir}/agents_outputs/<first_patient_id>/organ.nii.gz")
lesion_nii = nib.Nifti1Image(lesion_mask.astype(np.uint8), scan_nii.affine, scan_nii.header)
nib.save(lesion_nii, "{output_dir}/agents_outputs/<first_patient_id>/lesion.nii.gz")
print("Shape match: OK, Binary: OK")
```
