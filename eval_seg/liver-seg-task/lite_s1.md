Skill — How to download and inspect the model (examples only — use any approach that works):
```python
# Option 1: Download nnU-Net pretrained model for Task029_LiTS
# The nnU-Net framework provides pretrained models that can be downloaded
# via the nnU-Net CLI or programmatically.
# Check: https://github.com/MIC-DKFZ/nnUNet for pretrained model instructions

# Option 2: Search HuggingFace for LiTS nnU-Net checkpoints
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search="nnunet lits liver tumor")
for m in models:
    print(m.modelId, m.tags)

# Inspect checkpoint structure after download
import os, json, torch
for root, dirs, files in os.walk(model_dir):
    for f in files:
        print(os.path.join(root, f))
# Look for: fold_N/checkpoint_*.pth, dataset.json, plans.json

# Verify label map includes liver (1) and liver tumor (2)
```
