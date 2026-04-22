Skill — How to download and inspect a FeTA-trained model (examples only — use any approach that works):
```python
# Download a FeTA-trained nnU-Net v2 checkpoint from HuggingFace
from huggingface_hub import snapshot_download
model_dir = snapshot_download(
    "REPO_ID",                          # e.g. a FeTA24 submission repo
    local_dir="{output_dir}/model/feta_nnunet")

# Inspect the checkpoint structure
import os
for root, dirs, files in os.walk(model_dir):
    for f in files:
        print(os.path.join(root, f))

# Verify the LABEL MAP is the 7-class FeTA scheme (not some other fetal atlas):
#   0 = Background
#   1 = Extra-axial CSF (eCSF)
#   2 = Grey matter / cortex (GM)
#   3 = White matter (WM)
#   4 = Lateral ventricles (LV)
#   5 = Cerebellum (CBM)
#   6 = Deep grey matter / thalamus+putamen (SGM)
#   7 = Brainstem (BS)
# If a candidate model uses a different label scheme (e.g., dHCP parcellator
# with dozens of regions, or a model without brainstem), you will need to
# remap it at inference time. Verify by reading the model's docs / dataset.json.
```
