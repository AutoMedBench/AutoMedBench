Skill — How to compare candidate denoising methods (examples only — use any approach that works):
```bash
# Check what's pip-installable right now (all candidates in the list
# are either pip packages or have pre-trained HF checkpoints):
pip show bm3d scikit-image huggingface_hub 2>/dev/null | grep -E "^(Name|Version)"

# Search HuggingFace for image-restoration checkpoints:
# (you can also browse https://huggingface.co/models?search=restormer )
python -c "
import requests
for q in ('restormer', 'swinir denoising', 'ldct denoising'):
    r = requests.get('https://huggingface.co/api/models',
                     params={'search': q, 'limit': 5})
    for m in r.json():
        print(q, '->', m['modelId'])
"
```

```python
# Sketch of a comparison pipeline — verify each candidate is runnable.
import numpy as np, time

# Use one patient's input as a tiny smoke test for every candidate.
# Do NOT read any private/reference.npy — you do not have access to that.
from pathlib import Path
pid = sorted(Path("{data_dir}").iterdir())[0].name
x = np.load(Path("{data_dir}")/pid/"{input_filename}").astype(np.float32)

# Normalize HU -> [0, 1] in a shared window for every candidate.
HU_MIN, HU_MAX = -1024.0, 3072.0
x01 = np.clip((x - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0).astype(np.float32)

# Candidate A: BM3D (classical)
import bm3d
from skimage.restoration import estimate_sigma
sig = float(estimate_sigma(x01))
t0 = time.time(); y_bm3d = bm3d.bm3d(x01, sigma_psd=sig).astype(np.float32); t_bm3d = time.time() - t0
print("BM3D", y_bm3d.shape, f"t={{t_bm3d:.1f}}s")

# Candidate B: scikit-image Non-Local Means (classical, no training)
from skimage.restoration import denoise_nl_means
t0 = time.time(); y_nlm = denoise_nl_means(x01, patch_size=5, patch_distance=6, h=sig, fast_mode=True).astype(np.float32); t_nlm = time.time() - t0
print("NLM", y_nlm.shape, f"t={{t_nlm:.1f}}s")

# Candidate C: Restormer via HuggingFace (DNN, pretrained only — NEVER fine-tune)
# from huggingface_hub import snapshot_download
# rmp = snapshot_download("RESTORMER_REPO_ID", local_dir="{output_dir}/model/restormer")
# import torch
# model = ...  # load Restormer architecture, .load_state_dict(ckpt), .eval(), .cuda()
# with torch.no_grad():
#     y = model(torch.from_numpy(x01).unsqueeze(0).unsqueeze(0).cuda())

# Record timings + qualitative notes in plan.md. Pick the strongest option
# you can actually run. If a candidate needs weights you cannot download,
# drop it and move on — NEVER train from scratch.
```

```markdown
# Comparison table template for plan.md:
| Method     | Type     | Weights   | Wall-clock | Notes |
|------------|----------|-----------|------------|-------|
| BM3D       | classical | n/a       | ...        | ...   |
| NLM        | classical | n/a       | ...        | ...   |
| Restormer  | DNN      | HF: REPO  | ...        | ...   |
```
