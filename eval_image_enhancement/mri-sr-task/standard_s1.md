Skill — How to compare candidate SR methods (examples only — use any approach that works):
```bash
# All 3 candidates are either in torch (pre-installed), Pillow (pre-installed),
# or downloadable pretrained weights from HuggingFace.
pip show torch Pillow scikit-image 2>/dev/null | grep -E "^(Name|Version)"
```

```python
# Sketch: smoke-test every candidate on one public patient.
# Do NOT read any private/reference.npy — you do not have access to that.
import numpy as np, time
from pathlib import Path

pid = sorted(Path("{data_dir}").iterdir())[0].name
lr = np.load(Path("{data_dir}") / pid / "{input_filename}").astype(np.float32)

# Candidate A: PyTorch bicubic (classical, baseline)
import torch, torch.nn.functional as F
t0 = time.time()
t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0)
out_bicubic = F.interpolate(t, scale_factor=2, mode="bicubic", align_corners=False).squeeze().numpy().astype(np.float32)
t_bicubic = time.time() - t0
print("bicubic", out_bicubic.shape, f"t={{t_bicubic:.2f}}s")

# Candidate B: PIL Lanczos (classical, sharper edges)
from PIL import Image
t0 = time.time()
hr_h, hr_w = lr.shape[0] * 2, lr.shape[1] * 2
img = Image.fromarray(lr, mode="F")
out_lanczos = np.asarray(img.resize((hr_w, hr_h), resample=Image.LANCZOS), dtype=np.float32)
t_lanczos = time.time() - t0
print("lanczos", out_lanczos.shape, f"t={{t_lanczos:.2f}}s")

# Candidate C: Pretrained Restormer (DNN, inference-only)
# from huggingface_hub import snapshot_download
# repo = snapshot_download("RESTORMER_REPO_ID", local_dir="{output_dir}/model/restormer")
# import torch
# model = ... # load weights, .eval(), .cuda()
# with torch.no_grad():
#     out_restormer = model(torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).cuda()).squeeze().cpu().numpy()
# NEVER call .train() or .backward() — inference only.

# Record timings + qualitative notes in plan.md, pick the strongest you can actually run.
```

```markdown
# Comparison table template for plan.md:
| Method | Type (classical/DNN) | Public weights | Wall-clock | Notes |
|--------|----------------------|----------------|------------|-------|
| bicubic | classical | n/a | ... | smooth baseline |
| Lanczos | classical | n/a | ... | sharper edges |
| Restormer | DNN | HF: REPO | ... | pretrained natural-image restorer |
```
