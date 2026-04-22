Skill — How to search and compare FeTA-compatible models (examples only — use any approach that works):
```python
# Search HuggingFace for FeTA-trained or fetal-brain multi-tissue models
import requests
for q in ["feta fetal brain segmentation", "fetal brain nnunet",
          "fetal tissue segmentation MONAI", "FeTA 2024"]:
    resp = requests.get("https://huggingface.co/api/models",
                        params={"search": q, "limit": 10})
    for model in resp.json():
        print(q, "→", model["modelId"], model.get("tags", []))

# Download model weights from a URL
import urllib.request
urllib.request.urlretrieve(
    "https://example.com/feta_checkpoint.zip",
    "{output_dir}/model/weights.zip")

# Check MONAI Model Zoo bundles
from monai.bundle import download
# Try: monai.bundle.download("bundle_name", bundle_dir="{output_dir}/model")
# Inspect: configs/inference.json, docs/labels.json for label maps
```

```markdown
# Comparison table template for plan.md:
| Model | Label scheme covers 7 FeTA classes? | Reported mean Dice | Download Size | Ease of Setup | Notes |
|-------|-------------------------------------|--------------------|---------------|---------------|-------|
| ...   | ...                                 | ...                | ...           | ...           | ...   |
```

CRITICAL — verify that every candidate model outputs the 7 FeTA tissue
classes in a way that can be remapped to the benchmark label scheme:
  0 = BG, 1 = eCSF, 2 = GM, 3 = WM, 4 = LV, 5 = CBM, 6 = SGM, 7 = BS.
A model missing any of these classes (e.g., no brainstem) will score 0 on
that tissue and pull the mean down. Prefer models trained directly on FeTA.
