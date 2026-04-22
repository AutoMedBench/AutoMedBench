Skill — How to download and inspect the model (examples only — use any approach that works):
```python
# Download MedFormer PanTS checkpoint from HuggingFace
from huggingface_hub import snapshot_download
model_dir = snapshot_download(
    "AbdomenAtlas/MedFormerPanTS",
    local_dir="{output_dir}/model/medformer_pants")

# Inspect the checkpoint structure
import os
for root, dirs, files in os.walk(model_dir):
    for f in files:
        print(os.path.join(root, f))

# Check for: model weights, config files, label map
# Verify the model has pancreas organ + pancreatic tumor labels
```
