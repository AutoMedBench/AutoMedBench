Skill — Prepare DRUNet for LDCT (examples only — use any approach that works):
```python
# DRUNet via deepinv library. Inference-only; pretrained weights auto-downloaded.
# pip install deepinv   (do this in S2)

import numpy as np, torch
import deepinv as dinv

# Load once
model = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download")
model = model.eval().cuda()

# HU window normalization (LDCT convention)
HU_MIN, HU_MAX = -1024.0, 3072.0
def hu_to_01(x):  return np.clip((x - HU_MIN) / (HU_MAX - HU_MIN), 0, 1).astype(np.float32)
def inv_to_hu(y): return (y * (HU_MAX - HU_MIN) + HU_MIN).astype(np.float32)

# Forward call — pass sigma explicitly (DRUNet expects noise level in [0,1] range)
# Example: sigma from skimage OR a fixed literature prior
from skimage.restoration import estimate_sigma
# sigma = float(estimate_sigma(x01))  # typical 0.005-0.02 for LDCT_High
# Or: sigma = 0.05  # a reasonable prior for moderate-to-heavy LDCT noise

# Single-patient inference
def denoise_one(noisy_hu):
    x01 = hu_to_01(noisy_hu)
    sigma = float(estimate_sigma(x01))
    t = torch.from_numpy(x01).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        y01 = model(t, sigma=sigma).squeeze().cpu().numpy()
    return inv_to_hu(y01)
```

```markdown
# plan.md template
# Method: DRUNet (deepinv `DRUNet(in_channels=1, out_channels=1, pretrained='download')`)
# Weights: auto-downloaded, public
# Pre-processing: HU → [0,1] via clinical window (-1024, +3072)
# Sigma: [describe your choice: fixed? estimate_sigma? air-region?]
# Inference: model(x, sigma=...) on each 512×512 slice
# Post-processing: invert [0,1] → HU, cast float32
# Output: enhanced.npy, shape (512, 512), float32, HU range
```
