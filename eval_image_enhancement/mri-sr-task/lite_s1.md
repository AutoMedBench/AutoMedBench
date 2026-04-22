Skill — Prepare Swin2SR for MRI 2× super-resolution (examples only — use any approach that works):
```python
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

REPO = "caidas/swin2SR-classical-sr-x2-64"
proc = AutoImageProcessor.from_pretrained(REPO)
model = Swin2SRForImageSuperResolution.from_pretrained(REPO).eval().cuda()

def super_resolve(lr_grayscale_01, target_shape=(720, 512)):
    """lr: float32 [0,1] shape (360, 256) → hr: float32 [0,1] shape (720, 512)."""
    lr_u8 = (np.clip(lr_grayscale_01, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(lr_u8, mode="L").convert("RGB")
    inputs = proc(pil, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs).reconstruction       # (1, 3, H', W')
    # Reduce RGB → grayscale
    gray = out.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
    gray = np.clip(gray, 0, 1)
    # Resize to exact target shape if needed
    if gray.shape != target_shape:
        import torch.nn.functional as F
        t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        t2 = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        gray = t2.squeeze().numpy().astype(np.float32)
    return gray
```

```markdown
# plan.md template
# Method: Swin2SR x2 (HF `caidas/swin2SR-classical-sr-x2-64`)
# Weights: public, auto-downloaded by transformers
# Pre-processing: grayscale [0,1] → uint8 → PIL "L" → RGB
# Inference: model(inputs).reconstruction
# Post-processing: 3-channel → mean-grayscale, resize to (720, 512), clip [0,1]
# Output: enhanced.npy, shape (720, 512), float32, [0,1]
```
