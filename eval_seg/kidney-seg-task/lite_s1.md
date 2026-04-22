Skill — How to download and inspect the model (examples only — use any approach that works):
```python
# Download the nnU-Net KiTS19 checkpoint from HuggingFace
from huggingface_hub import snapshot_download
model_dir = snapshot_download(
    "KagglingFace/nnUNet-KiTS19-3d-lowres-50epochs",
    local_dir="{output_dir}/model/nnunet_kits19")

# Inspect the checkpoint structure
import os, json, torch
for root, dirs, files in os.walk(model_dir):
    for f in files:
        print(os.path.join(root, f))
# Look for: fold_N/checkpoint_*.pth, dataset.json, plans.json

# Load and inspect label map from dataset.json or plans.json
ckpt = torch.load(os.path.join(model_dir, "fold_1", "checkpoint"),
                   map_location="cpu")
# Check ckpt keys: 'trainer_name', 'init_args', 'network_weights'

# nnU-Net inference (no env vars needed for inference-only):
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# Create predictor and load model — check nnUNetPredictor docs for args
```
