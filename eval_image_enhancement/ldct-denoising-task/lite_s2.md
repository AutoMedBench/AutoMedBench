Skill — Environment + DRUNet load (examples only — use any approach that works):
```bash
python -m venv --system-site-packages $WORKSPACE_DIR/env
source $WORKSPACE_DIR/env/bin/activate
pip install -r $(python -c "import sys; print('REQS_PATH')")   # or use requirements.txt
pip install deepinv   # if not already in requirements.txt
```

```python
# Smoke test: verify DRUNet loads + forward-pass works on GPU
import numpy as np, torch
import deepinv as dinv

model = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download").eval().cuda()
dummy = torch.randn(1, 1, 512, 512).cuda()
with torch.no_grad():
    out = model(dummy, sigma=0.05)
print("OK — DRUNet output:", out.shape, out.dtype, out.device)
# Expected: torch.Size([1, 1, 512, 512]) float32 cuda:0

# REMINDER: inference-only. No .train(), no optimizer, no .backward().
```
