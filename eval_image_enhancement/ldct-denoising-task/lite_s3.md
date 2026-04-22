Skill — How to validate on one patient (examples only — use any approach that works):
```python
import os, numpy as np, bm3d
from skimage.restoration import estimate_sigma

HU_MIN, HU_MAX = -1024.0, 3072.0

# Pick the first patient in the data dir
patients = sorted(os.listdir("{data_dir}"))
pid = patients[0]
inp = np.load(os.path.join("{data_dir}", pid, "{input_filename}")).astype(np.float32)
print(f"input {pid}: shape={inp.shape}, dtype={inp.dtype}, HU range=[{{inp.min():.1f}}, {{inp.max():.1f}}]")

# HU -> [0, 1]
x01 = np.clip((inp - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0).astype(np.float32)
sigma = float(estimate_sigma(x01))
print(f"estimated sigma_psd in [0,1] space: {{sigma:.4f}}")

# BM3D (2D, classical, no training)
y01 = bm3d.bm3d(x01, sigma_psd=sigma).astype(np.float32)
out = (y01 * (HU_MAX - HU_MIN) + HU_MIN).astype(np.float32)

# Sanity checks
assert out.shape == inp.shape, f"shape mismatch: out={{out.shape}} vs inp={{inp.shape}}"
assert out.dtype == np.float32
assert not np.isnan(out).any() and not np.isinf(out).any()

print(f"output: shape={{out.shape}}, HU range=[{{out.min():.1f}}, {{out.max():.1f}}]")
print(f"mean |inp - out| (HU) = {{np.abs(inp - out).mean():.2f}}")
print(f"input std = {{inp.std():.2f}}, output std = {{out.std():.2f}}")
# Expect: out std slightly lower than inp std (noise reduced), but not drastically.
# Mean absolute diff in HU should be > 0 but not huge (>200 HU would indicate over-smoothing).
```
