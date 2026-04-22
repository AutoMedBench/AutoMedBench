Skill — Validate your chosen SR method on 1 patient (examples — use any approach that works):
```python
import os, numpy as np

pid = sorted(os.listdir("{data_dir}"))[0]
lr = np.load(os.path.join("{data_dir}", pid, "{input_filename}")).astype(np.float32)
print(f"validating on {{pid}}: LR={{lr.shape}} dtype={{lr.dtype}} range=[{{lr.min():.3f}}, {{lr.max():.3f}}]")

# Run your chosen SR pipeline:
out = run_your_sr(lr)  # <- replace with your actual call; returns (720, 512) float32

# Sanity checks — all must pass before S4:
expected_shape = (lr.shape[0] * 2, lr.shape[1] * 2)
assert out.shape == expected_shape, f"shape mismatch: {{out.shape}} vs {{expected_shape}}"
assert out.dtype == np.float32, f"expected float32, got {{out.dtype}}"
assert not np.isnan(out).any() and not np.isinf(out).any()
assert out.min() >= -0.1 and out.max() <= 1.1, f"range looks wrong: [{{out.min():.3f}},{{out.max():.3f}}]"

# Heuristic: output should keep overall brightness of input
mean_diff = abs(out.mean() - lr.mean())
print(f"out mean={{out.mean():.3f}}, lr mean={{lr.mean():.3f}}, diff={{mean_diff:.3f}}")
# Expected: mean_diff < 0.05 (meaningful upsample preserves mean intensity)
# If out.std() << lr.std(), the method is over-smoothing.

# REMINDER: inference-only. No optimizer, no .backward(), no .train().
```
