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

Skill — How to validate on one patient (examples only — use any approach that works):
```python
import nibabel as nib
import numpy as np

# Load input scan
scan_path = "{data_dir}/<first_patient_id>/{input_filename}"
scan_nii = nib.load(scan_path)
scan_data = scan_nii.get_fdata()
print(f"Scan shape: {scan_data.shape}, dtype: {scan_data.dtype}")

# Run inference (model-specific) — ON GPU
# ... your inference code here ...
organ_mask = ...   # numpy array, same shape as scan_data
lesion_mask = ...  # numpy array, same shape as scan_data

# Verify output
assert organ_mask.shape == scan_data.shape, f"Shape mismatch"
assert set(np.unique(organ_mask)).issubset({0, 1}), f"Not binary"
assert organ_mask.sum() > 1000, f"Organ mask too small"

# CRITICAL: Check lesion output
lesion_ratio = lesion_mask.sum() / max(organ_mask.sum(), 1)
print(f"Organ voxels: {organ_mask.sum()}, Lesion voxels: {lesion_mask.sum()}")
print(f"Lesion/Organ ratio: {lesion_ratio:.4f}")
if lesion_ratio == 0.0:
    print("WARNING: Lesion mask is EMPTY. Model is NOT detecting tumors.")
else:
    print(f"Lesion check: OK (ratio={lesion_ratio:.4f})")

# Save
organ_nii = nib.Nifti1Image(organ_mask.astype(np.uint8), scan_nii.affine, scan_nii.header)
nib.save(organ_nii, "{output_dir}/agents_outputs/<first_patient_id>/organ.nii.gz")
lesion_nii = nib.Nifti1Image(lesion_mask.astype(np.uint8), scan_nii.affine, scan_nii.header)
nib.save(lesion_nii, "{output_dir}/agents_outputs/<first_patient_id>/lesion.nii.gz")
```
