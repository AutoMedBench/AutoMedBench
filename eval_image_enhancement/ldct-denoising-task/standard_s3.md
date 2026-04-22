Skill — Validating your chosen method on one patient (examples — use any approach that works):
```python
import os, numpy as np

pid = sorted(os.listdir("{data_dir}"))[0]
inp = np.load(os.path.join("{data_dir}", pid, "{input_filename}")).astype(np.float32)
print(f"validating on {{pid}}: inp shape={{inp.shape}} dtype={{inp.dtype}} HU=[{{inp.min():.0f}},{{inp.max():.0f}}]")

# Run your chosen enhancement pipeline end-to-end on `inp`.
# (The exact call depends on what you picked in S1.)
out = run_your_method(inp)  # <- replace with your actual call

# Sanity checks — all must pass before S4:
assert out.shape == inp.shape, f"shape mismatch: {{out.shape}} vs {{inp.shape}}"
assert out.dtype == np.float32, f"expected float32, got {{out.dtype}}"
assert not np.isnan(out).any() and not np.isinf(out).any(), "NaN/Inf in output"
assert out.min() >= -2000 and out.max() <= 4000, f"HU range looks wrong: [{{out.min():.0f}},{{out.max():.0f}}]"

# Make sure the method is actually doing something but not wrecking the image.
# You do NOT have access to a reference — only compare to the input.
diff_mean = np.abs(inp - out).mean()
ratio_std = out.std() / max(inp.std(), 1e-6)
print(f"mean|inp-out|={{diff_mean:.2f}} HU, output/input std ratio={{ratio_std:.3f}}")
# Heuristics to interpret:
#   diff_mean ≈ 0 and ratio_std ≈ 1.0  -> method is a no-op (bad)
#   diff_mean > 300 HU                  -> method is probably over-smoothing (bad)
#   ratio_std < 0.3                     -> severe over-smoothing (bad)
#   diff_mean in 5-100 HU, ratio 0.7-0.95 -> plausible denoising (good)

# REMINDER: do NOT create optimizers, do NOT call .backward(), do NOT
# call model.train(). This is inference-only.
```
