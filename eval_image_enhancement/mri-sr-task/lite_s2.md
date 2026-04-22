Skill — Environment + Swin2SR load (examples only — use any approach that works):
```bash
python -m venv --system-site-packages $WORKSPACE_DIR/env
source $WORKSPACE_DIR/env/bin/activate
pip install -r $(python -c "import sys; print('REQS_PATH')")
pip install transformers  # if not already in requirements
```

```python
# Smoke test: verify Swin2SR loads + forward-pass works
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

REPO = "caidas/swin2SR-classical-sr-x2-64"
proc = AutoImageProcessor.from_pretrained(REPO)
model = Swin2SRForImageSuperResolution.from_pretrained(REPO).eval().cuda()
dummy = np.random.rand(256, 256).astype(np.float32)
pil = Image.fromarray((dummy * 255).astype(np.uint8)).convert("RGB")
inputs = {k: v.cuda() for k, v in proc(pil, return_tensors="pt").items()}
with torch.no_grad():
    out = model(**inputs).reconstruction
print("OK — Swin2SR output:", out.shape, out.device)
# Expected: torch.Size([1, 3, ~512, ~512]) cuda:0

# REMINDER: inference-only. No .train(), no optimizer, no .backward().
```
