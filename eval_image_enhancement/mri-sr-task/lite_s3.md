Skill — Validate on 1 patient (examples — use any approach that works):
```python
import os, numpy as np, torch
import torch.nn.functional as F

# Pick the first patient
pids = sorted(os.listdir("{data_dir}"))
pid = pids[0]
lr = np.load(os.path.join("{data_dir}", pid, "{input_filename}")).astype(np.float32)
print(f"{pid}: LR shape={lr.shape} dtype={lr.dtype} range=[{lr.min():.3f}, {lr.max():.3f}]")

# Bicubic 2x upsample
t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0)
hr = F.interpolate(t, scale_factor=2, mode="bicubic", align_corners=False)
out = hr.squeeze().numpy().astype(np.float32)

# Clamp back to [0, 1] since bicubic can slightly overshoot
out = np.clip(out, 0.0, 1.0).astype(np.float32)

# Sanity checks
assert out.shape == (lr.shape[0]*2, lr.shape[1]*2), f"shape mismatch: {{out.shape}}"
assert out.dtype == np.float32
assert not np.isnan(out).any() and not np.isinf(out).any()
assert 0.0 <= out.min() and out.max() <= 1.0

print(f"output: shape={{out.shape}} range=[{{out.min():.3f}}, {{out.max():.3f}}]  std={{out.std():.3f}}")
# Expected: (720, 512), no NaN/Inf, range [0, 1], std ~0.15-0.30 for MRI brain slices
```
